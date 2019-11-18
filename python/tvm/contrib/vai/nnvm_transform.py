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

""" Vitis-AI NNVM partitioning for Xilinx FPGA acceleration """

import nnvm
import nnvm.symbol as sym

# RECURSIVE ALGORITHM TO PARSE THE GRAPH,
# AND FIND NODES BETWEEN THE OUTPUTS AND INPUTS
# PROVIDED BY THE COMPILER
def fuse(graph, xfuse_inputs, input_list,queue, fuse_list, count, platform):

    # RETURN CONDITION
    if not queue:
        return
    
    nid = queue.pop()
    
    fuse_list.append(nid)
    count = count + 1
    children = [e[0] for e in graph[nid]["inputs"]]
    #if the nid is not visited then you could queue
    for nid in children:
        inputs = graph[nid]["inputs"]
        attrs = graph[nid].get("attrs", {})
        node_name = graph[nid]["name"]
        op_name = graph[nid]["op"]
        if node_name in input_list:
            #TODO: nid-1 may not be a good candidate for finding the input node
            if platform == 'DPU':
                fuse_list.append(nid)
                xfuse_inputs.append(nid-1)
            else:
                xfuse_inputs.append(nid)
        elif nid in fuse_list:
            continue
        else:
            #fuse_list.append(nid)
            queue.append(nid)


    fuse(graph,xfuse_inputs,input_list,queue,fuse_list,count, platform)
    

# FUNCTION TO PARSE THROUGH THE NNVM GRAPH,
# TRIM NODES BASED ON THE OUTPUT OF EXTERNAL COMPILER,
# AND CREATE A NEW NNVM GRAPH
def graph_reconst(path, nnvm_graph, output_layout, model_name, output_layers=None, platform = 'XDNN'): 
    node_map={}
    accel_inputs = []

    if platform == 'DPU':
        compiler_json_file = path + "/dpu_xgraph.json"
        dnn_name           = path + "/dnnc_comp_xp0.json"
        with open(compiler_json_file) as json_file:
            json_graph = json.load(json_file)
        with open(dnn_name) as json_file:
            dnnc_comp_d = json.load(json_file)
        
        # TEMP

        for node in json_graph['nodes']:
            if node['LayerParameter']['type'][0] == 'DPU':
                kernel_name           = node['name']
                input_names           = node['LayerParameter']['attrs']['input_names']
                output_names          = node['LayerParameter']['attrs']['output_names']
                graph_inputs          = node['LayerParameter']['attrs']['input_layers'][input_names[0]]
                graph_outputs         = [node['LayerParameter']['attrs']['output_layers'][output_names[0]][-1]]
                compiler_shape_output = node['LayerParameter']['shapes'] 
                
    else:
        compiler_json_file = path + "/work/" + model_name + "_compiler.json"
        with open(compiler_json_file) as json_file:
            json_graph = json.load(json_file)
        
        graph_inputs          = json_graph["inputs"]
        graph_outputs         = json_graph["outputs"]                                         
        compiler_shape_output = json_graph["network"][-1]["outputshapes"]
        
    xfuse_inputs=[]
    fuse_list=[]
    queue=[]


    if platform == 'DPU':
        input_list = graph_inputs
    else:
        input_list = [n['input_name'] for n in graph_inputs]
        
    for layer in graph_outputs:

        if platform == 'DPU':
            layer_name = layer
        else:
            layer_name = layer['previous_layers'][0]
            
        # PARSE THROUGH THE GRAPH AND FIND THE MATCHING OUTPUT NODE
        layer_nid=0
        for nid, node in enumerate(nnvm_graph): 
            node_name = node["name"]
            if layer_name == node_name:
                layer_nid=nid
        assert layer_nid != 0, "The output node was not found"
        queue.append(layer_nid)
        fuse(nnvm_graph,xfuse_inputs,input_list,queue,fuse_list,0, platform)

    # GRAPH RECONSTRUCTION
    for nid, node in enumerate(nnvm_graph):
        inputs = node["inputs"]
        attrs = node.get("attrs", {})
        node_name = node["name"]
        op_name = node["op"]
        get_clone = lambda c, o_n, n_n, a: getattr(nnvm.symbol, o_n)(
            *c, name=n_n, **a)
        new_entry = None
        if nid in fuse_list:
            for layer in graph_outputs:
                if platform == 'DPU':
                    layern_name = layer
                else:
                    layer_name = layer['previous_layers'][0]
                if node_name == layer_name:
                    # CREATE ACCEL NODE
                    if platform == 'XDNN' and output_layout == 'NHWC':
                        output_shape = (1,
                                        compiler_shape_output[2],
                                        compiler_shape_output[3],
                                        compiler_shape_output[1])
                        
                    elif platform == 'DPU'  and output_layout == 'NCHW':
                        output_shape = (1,
                                        compiler_shape_output[3],
                                        compiler_shape_output[1],
                                        compiler_shape_output[2])
                    else: 
                        output_shape = (1,
                                        compiler_shape_output[1],
                                        compiler_shape_output[2],
                                        compiler_shape_output[3])   

                    new_entry = sym.accel(*accel_inputs,
                                          path          = path,
                                          kernel_name   = kernel_name,
                                          input_names   = dnnc_comp_d[input_names[0] ],
                                          output_names  = dnnc_comp_d[output_names[0]],
                                          output_shape  = output_shape,
                                          output_layout = output_layout,
                                          model_name    = model_name,
                                          platform      = platform)

                    node_map[nid] = new_entry
        else:
                    
            if op_name == "null":
                new_entry = nnvm.symbol.Variable(node_name)
                accel_inputs.append(new_entry)
            else:
                children = [node_map[e[0]] for e in node["inputs"]]
                new_entry = get_clone(children, op_name, node_name, attrs)
                  
            node_map[nid] = new_entry

    # ADD ANY LAYERS NECESSARY AT THE END BASED ON THE OUTPUT_LAYERS LIST
    if output_layers:
        for layer in output_layers:
            if layer =='Softmax':
                nodes = list(node_map.keys())
                node_map[nodes[-1]+1] = sym.softmax(node_map[nodes[-1]])
                
    node_map_list = list(node_map.items())

    # ASSUMING THE LAST NODE IS ALWAYS THE OUTPUT
    return nnvm.graph.create(node_map_list[-1][1])