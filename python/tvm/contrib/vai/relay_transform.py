# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

""" 
Vitis-AI Relay pass to partition Relay graph for Xilinx FPGA acceleration 

"""

import os
import json
import tvm
from tvm import relay

import xfgraph
# Register dpu as a xfgraph target device
from xfgraph.contrib import dnndk

from xfgraph.frontend import from_relay
from xfgraph.graph.io.xgraph_io import XGraphIO
from xfgraph.generator.tensorflow import XfGraphTfGeneratorOptimizer
from xfgraph.contrib.dnndk.decent_quantizer import DECENTQuantizer
from xfgraph.contrib.dnndk.dnnc_compiler import DNNCCompiler

@relay.transform.module_pass(opt_level=4)
class PartitioningPass:

    """
    The Vitis-AI partitioning pass

    Arguments
    ---------
    target: str
        the target accelerator, only 'dpu' is supported at the moment

    params: dict from str to array
        the relay model parameters

    inputs_func: function
        a python function which takes an iterator number and a layout and 
        provides a numpy array of inputs to be used for quantization 
        calibration

    layout: str
        the layout of the Relay model, only 'NCHW' and 'NHWC' supported 
        at the moment
    
    """

    def __init__(self, target, params, inputs_func, layout):

        if target not in ['dpu', 'dpu_nocompile']: # TODO: remove dpu_nocompile
            raise ValueError("Invalid target: {} for the Vitis-AI"\
                " partitioning pass, only 'dpu' target is supported"\
                " at the moment.".format(target))

        if layout not in ['NCHW', 'NHWC']:
            raise ValueError("Invalid layout: {} for Vitis-AI partitioning"\
                " pass, only 'NCHW' and 'NHWC' are supported at the moment."
                .format(target))

        self.target = target
        self.params = params
        self.inputs_func = inputs_func
        self.layout = layout

        self.work_dir = '/tmp/vai'
        os.makedirs(self.work_dir, exist_ok=True)
        
    def transform_module(self, mod, ctx):
        
        target = self.target

        # TODO params can be found in ctx??
        xfgraph = from_relay(mod, self.params, data_layout=self.layout)

        if target == 'dpu_nocompile':
            target = 'dpu'
        elif target == 'dpu':

            # Optimize xfgraph for Tensorflow generation
            xfgraph.optimize(XfGraphTfGeneratorOptimizer)
            
            # Internal partitioning
            xfgraph.partition(devices=[target])

            dpu_xgraph = xfgraph.schedule(device='dpu')
            XGraphIO.save(dpu_xgraph, os.path.join(self.work_dir, 'dpu_xgraph'))

            # Quantization
            quantizer = DECENTQuantizer(xfgraph, inputs_func, self.work_dir)
            netcfgs = quantizer.quantize(subgraphs_only=True)

            # Compilation
            #dcf = "/dnndk/dcf/ZCU104.dcf"
            dcf = "/dnndk/dcf/Ultra96.dcf"
            
            compiler = DNNCCompiler(xfgraph, netcfgs=netcfgs, dcf=dcf)
            compiler.compile()

        else:
            raise ValueError("Unsupported target: {}".format(target))

        # Relay partitioning
        mod = self.reconst_graph(
            mode=mod,
            path=self.work_dir,
            output_layout=self.layout,           
            model_name="model_name?",
            output_layers=[]
        )

        return mod


    def reconst_graph(self, mod, path, output_layout, model_name, platform, output_layers=None):

        #pdb.set_trace()
        node_map={}
        xdnn_inputs = []

        if platform == 'DPU':
            compiler_json_file = path + "/dpu_xgraph.json"
            dnn_name           = path + "/dnnc_comp_xp0.json"
            with open(compiler_json_file) as json_file:
                json_graph = json.load(json_file)
            with open(dnn_name) as json_file:
                dnnc_comp_d = json.load(json_file)
        
            for node in json_graph['nodes']:
                if node['LayerParameter']['type'][0] == 'DPU':
                    kernel_name           = node['name']
                    input_names           = node['LayerParameter']['attrs']['input_names']
                    output_names          = node['LayerParameter']['attrs']['output_names']
                    graph_inputs          = node['LayerParameter']['attrs']['input_layers'][input_names[0]]
                    graph_outputs         = [node['LayerParameter']['attrs']['output_layers'][output_names[0]][-1]]
                    compiler_shape_output = node['LayerParameter']['shapes'] 
        else:
            compiler_json_file = path + "/work/" +  model_name + "_compiler.json"
            with open(compiler_json_file) as json_file:
                json_graph = json.load(json_file)
        
            graph_inputs  = json_graph["inputs"]
            graph_outputs = json_graph["outputs"]
            

            #pdb.set_trace()
            compiler_shape_output = json_graph["network"][-1]["outputshapes"]

            
        xfuse_inputs=[]
        fuse_list=[]
        queue=[]

        if platform == 'DPU':
            input_list = graph_inputs
        else:
            input_list = [self.extract_hash(n,'input_name') for n in graph_inputs]

        expr = mod.functions[mod.get_global_var('main')]
        expr = expr.body
        #traverse(expr)
        for output in graph_outputs:
            # TODO: PREVIOUS_LAYERS IS NOT A GOOD CHOICE OF GETTING THE OUTPUT NAME
            # WILL NEED TO CHANGE LATER
            output_hash = self.extract_hash(output,'previous_layers')
            expr = self.traverse(expr,output_hash, input_list, output_layout, compiler_shape_output, path, model_name)
        
        #pdb.set_trace()
        # ADD ANY LAYERS NECESSARY AT THE END BASED ON THE OUTPUT_LAYERS LIST
        if output_layers:
            for layer in output_layers:
                if layer =='Softmax':
                    expr = relay.nn.softmax(expr)
    
        mod = relay.Module.from_expr(expr)

        return mod

    def extract_hash(self,name, key):
        #pdb.set_trace()
        if key == 'input_name':
            val = name[key].split('-')
        else:
            val = name[key][0].split('-')
            
        try:
            return int(val[0])
        except (ValueError):
            #pdb.set_trace()
            if len(val) == 1:
                return val[0]
            else:
                return int(val[1])



    # TODO: NEED TO CREATE AN ARRAY TO RETURN
    # WHEN THERE ARE MULTIPLE INPUTS
    def recurse(self, expr, input_list):
        #pdb.set_trace()
        print(type(expr))
        if (isinstance(expr,  tvm.relay.expr.Function)):
            print (expr.params)
            print (expr.body)
            return self.recurse(expr.body, name)
        
        elif (isinstance(expr, tvm.relay.expr.Call)):
            print(expr.op)
            if (hash(expr) in input_list):
                print("returning %s expr" %(expr.op))
                return expr
            for node in expr.args:
                ret = self.recurse(node, input_list)
                if ret is not None:
                    return ret
            return None
        
        elif (isinstance(expr,tvm.relay.expr.TupleGetItem)):
            #pdb.set_trace()
            if (hash(expr) in input_list):
                print("returning %s expr" %(expr.index))
                return expr
            return self.recurse(expr.tuple_value, input_list)
        
        elif (isinstance(expr,tvm.relay.expr.Var)):
            print(expr.name_hint)
            try:
                input_name = int(expr.name_hint)
            except (ValueError):
                if expr.name_hint == 'data':
                    pdb.set_trace()
                    input_name = 'data'
                else:
                    input_name = None

            if (hash(expr) in input_list or input_name in input_list):
                print("returning %s expr" %(expr.name_hint))
                return expr
            else:
                return None
        elif (isinstance(expr,tvm.relay.expr.Tuple)):
            if (hash(expr) in input_list):
                return expr
            for node in expr.fields:
                ret = self.recurse(node, input_list)
                if ret is not None:
                    return ret
                else:
                    return None    
            
        else:
            #pdb.set_trace()
            print("Missing condition to handle node type %s", type(expr))


        
    def traverse(self,expr, output_hash, input_list, output_layout, output_shape, path, model_name):
    
        #assert (isinstance(expr, tvm.relay.expr.Call))
        if (hash(expr) == output_hash):
            print("--debug: found ouptut name")
            for node in expr.args:
                input_node = self.recurse(node,input_list)
                if(input_node is not None):
                    print("--debug: found input node")
                    #pdb.set_trace()

                    if output_layout == 'NHWC':
                        output_shape = (1,output_shape[2],output_shape[3],output_shape[1])
                    else: #DEFAULT CASE IS ASSUMED TO BE 'NCHW'
                        output_shape = (1,output_shape[1],output_shape[2],output_shape[3])   

                    op = relay.nn.accel([input_node],output_layout=output_layout,path=path,model_name = model_name, output_shape=output_shape)
                
                    return op
                
            return None
            

        else:
            if isinstance(expr,tvm.relay.expr.Constant):
                return None

            elif (isinstance(expr, tvm.relay.expr.Call)):
                for node in expr.args:
                    output_node = self.traverse(node,output_hash,input_list, output_layout, output_shape, path, model_name)
                    if (output_node is not None):
                        break
                # CASES WHERE THE OUTPUT_NODE IS NOT IN THE CHOSEN BRANCH OF THE GRAPH
                if output_node is None:
                    return output_node
                
            elif(isinstance(expr,tvm.relay.expr.TupleGetItem)):
                return traverse(expr.tuple_value, output_hash,input_list,output_layout,output_shape, path, model_name)
        
            elif(isinstance(expr,tvm.relay.expr.Tuple)):
                return None
            #return traverse(expr.fields, output_hash,input_list,output_layout,output_shape)
            elif(isinstance(expr,tvm.relay.expr.Var)):
                return None
            else:
                print("-- Warning: Encoutered an unknown node type %s while traversing the graph",type(arg))
                return None


            
            #RECONSTRUCT THE GRAPH BY RECREATING THE NON-PARTITIONED NODES
            if (isinstance(expr, tvm.relay.expr.Call)):
            
                children=[]
                for node in expr.args:

                    if (isinstance(node, tvm.relay.expr.Call)):
                        if (node.op == output_node.op or
                            output_node.op.name == 'nn.accel'):
                            children.append(output_node)
                        else:
                            children.append(node)
                    else:
                        children.append(node)
                    
                new_node = relay.Call(expr.op,children,expr.attrs,expr.type_args)

            else:
                assert isinstance(expr, tvm.relay.expr.Call), print("Condition to reconstruct node type %s has not been implemented", type(expr) )

         
            
        return new_node