import json
import pdb
import nnvm
import nnvm.symbol as sym
import os
import tvm
from tvm import relay

# RECURSIVE ALGORITHM TO PARSE THE GRAPH,
# AND FIND NODES BETWEEN THE OUTPUTS AND INPUTS
# PROVIDED BY THE COMPILER
def fuse(graph, xfuse_inputs, input_list,queue, fuse_list, count):

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
        #children = [e[0] for e in nid["inputs"]]
        attrs = graph[nid].get("attrs", {})
        node_name = graph[nid]["name"]
        op_name = graph[nid]["op"]
        if node_name in input_list:
            # Don't add the node to trim list
            xfuse_inputs.append(nid)
        elif nid in fuse_list:
            continue
        else:
            #fuse_list.append(nid)
            queue.append(nid)


    fuse(graph,xfuse_inputs,input_list,queue,fuse_list,count)
    

# FUNCTION TO PARSE THROUGH THE NNVM GRAPH,
# TRIM NODES BASED ON THE OUTPUT OF EXTERNAL COMPILER,
# AND CREATE A NEW NNVM GRAPH
def graph_reconst(path, nnvm_graph, output_layout, model_name, output_layers=None): 
    node_map={}
    accel_inputs = []

    compiler_json_file = path + model_name + "_compiler.json"
    with open(compiler_json_file) as json_file:
        json_graph = json.load(json_file)
        
    graph_inputs = json_graph["inputs"]
    graph_outputs = json_graph["outputs"]
    graph_outputs = json_graph["outputs"]
    #pdb.set_trace()
    compiler_shape_output = json_graph["network"][-1]["outputshapes"]
    xfuse_inputs=[]
    fuse_list=[]
    queue=[]
    input_list = [n['input_name'] for n in graph_inputs]
    for layer in graph_outputs:
        layer_name = layer['previous_layers'][0]
        # PARSE THROUGH THE GRAPH AND FIND THE MATCHING OUTPUT NODE
        layer_nid=0
        for nid, node in enumerate(nnvm_graph): # have to change that later
            node_name = node["name"]
            if layer_name == node_name:
                layer_nid=nid
        assert layer_nid != 0, "The output node was not found"
        queue.append(layer_nid)
        fuse(nnvm_graph,xfuse_inputs,input_list,queue,fuse_list,0)

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
                layer_name = layer['previous_layers'][0]
                if node_name == layer_name:
                    # CREATE ACCEL NODE
                    if output_layout == 'NHWC':
                        output_shape = (1,compiler_shape_output[2],compiler_shape_output[3],compiler_shape_output[1])
                    else: #DEFAULT CASE IS ASSUMED TO BE 'NCHW'
                        output_shape = (1,compiler_shape_output[1],compiler_shape_output[2],compiler_shape_output[3])   

                    new_entry = sym.accel(*accel_inputs, path=path, output_shape=output_shape, output_layout = output_layout, model_name = model_name)
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
    #return nnvm.graph.create(node_map_list[1][1])

    

'''
def traverse(expr, output_hash, input_list, output_layout, output_shape):
    
    # TODO: NEED TO CREATE AN ARRAY TO RETURN
    # WHEN THERE ARE MULTIPLE INPUTS
    print("Traverse %s",type(expr))
    if (isinstance(expr, tvm.relay.expr.Call)):
        print(expr.op)
        print(len(expr.args))
        
    #pdb.set_trace()
    def recurse(expr, input_list):
        #pdb.set_trace()
        print(type(expr))
        if (isinstance(expr,  tvm.relay.expr.Function)):
            print (expr.params)
            print (expr.body)
            return recurse(expr.body, name)
        
        elif (isinstance(expr, tvm.relay.expr.Call)):
            print(expr.op)
            if (hash(expr) in input_list):
                print("returning %s expr" %(expr.op))
                return expr
            for arg in expr.args:
                ret = recurse(arg, input_list)
                if ret is not None:
                    return ret
            return None
        
        elif (isinstance(expr,tvm.relay.expr.TupleGetItem)):
            if (hash(expr) in input_list):
                print("returning %s expr" %(expr.index))
                return expr
            return recurse(expr.tuple_value, input_list)
        
        elif (isinstance(expr,tvm.relay.expr.Var)):
            print(expr.name_hint)
            if (hash(expr) in input_list or int(expr.name_hint) in input_list):
                print("returning %s expr" %(expr.name_hint))
                return expr
            else:
                return None
        else:
            #pdb.set_trace()
            print("Missing condition to handle node type %s", type(expr))

    #assert (isinstance(expr, tvm.relay.expr.Call))
    if (hash(expr) == output_hash):
        print("--debug: found ouptut name")
        for arg in expr.args:
            input_node = recurse(arg,input_list)
            if(input_node is not None):
                print("--debug: found input node")
                #pdb.set_trace()

                if output_layout == 'NHWC':
                    output_shape = (1,output_shape[2],output_shape[3],output_shape[1])
                else: #DEFAULT CASE IS ASSUMED TO BE 'NCHW'
                    output_shape = (1,output_shape[1],output_shape[2],output_shape[3])   
                    
                op = relay.nn.xdnn([input_node],output_layout=output_layout,path=os.getcwd(),output_shape=output_shape)
                
                return op
            else: 
                return None
            

    else:
        if isinstance(expr,tvm.relay.expr.Constant):
            return None

        elif (isinstance(expr, tvm.relay.expr.Call)):
            for arg in expr.args:
                output_node = traverse(arg,output_hash,input_list, output_layout, output_shape)
                if (output_node is not None):
                    break
            # CASES WHERE THE OUTPUT_NODE DOES NOT MERGE INTO THE INPUT_NODE
            if output_node is None:
                return output_node
            
        elif(isinstance(expr,tvm.relay.expr.TupleGetItem)):
             return traverse(expr.tuple_value, output_hash,input_list,output_layout,output_shape)
        
        elif(isinstance(expr,tvm.relay.expr.Tuple)):
            return None
            #return traverse(expr.fields, output_hash,input_list,output_layout,output_shape)
        elif(isinstance(expr,tvm.relay.expr.Var)):
            return None
        else:
            #print(expr.fields)
            assert False, print("Missing condition to handle node type %s", type(expr))

   
        if (isinstance(expr, tvm.relay.expr.Call)):
            
            children=[]
            for arg in expr.args:
                
                #  or output_node.op == xdnn): #TODO: HAVE TO FIUGREOUT LATER
                if (isinstance(arg, tvm.relay.expr.Call)):
                    
                    if (arg.op == output_node.op or
                        output_node.op.name == 'nn.xdnn'):
                        children.append(output_node)
                    else:
                        children.append(arg)
                else:
                    children.append(arg)
                                            
                #assert isinstance(expr, tvm.relay.expr.Call), print("No instruction to reconstruct node type %s", type(expr) )

               
                    
            new_node = relay.Call(expr.op,children,expr.attrs,expr.type_args)

        else:
            #pdb.set_trace()
            assert isinstance(expr, tvm.relay.expr.Call), print("Condition to reconstruct node type %s has not been implemented", type(expr) )

         
            
        return new_node

def extract_hash(name, key):
    if key == 'input_name':
        val = name[key].split('-')
    else:
        val = name[key][0].split('-')

    try:
        return int(val[0])
    except (ValueError):
        return int(val[1])
        

    
# Add any extra operations to the graph
def reconst_graph(mod, path, output_layout, output_layers=None):
    input_name = 'nn.conv2d'
    output_name = 'nn.relu'

    node_map={}
    xdnn_inputs = []

    with open(path) as json_file:
        json_graph = json.load(json_file)
        
    graph_inputs  = json_graph["inputs"]
    graph_outputs = json_graph["outputs"]

    #pdb.set_trace()
    compiler_shape_output = json_graph["network"][-1]["outputshapes"]
    xfuse_inputs=[]
    fuse_list=[]
    queue=[]
    #input_list = [n['input_name'] for n in graph_inputs]
    input_list = [extract_hash(n,'input_name') for n in graph_inputs]
    
    
    #pdb.set_trace()
    expr = mod.functions[mod.get_global_var('main')]
    expr = expr.body
    #traverse(expr)
    for output in graph_outputs:
        # TODO: PREVIOUS_LAYERS IS NOT A GOOD CHOICE OF GETTING THE OUTPUT NAME
        # WILL NEED TO CHANGE LATER
        output_hash = extract_hash(output,'previous_layers')
        expr = traverse(expr,output_hash, input_list, output_layout, compiler_shape_output)
        
    #pdb.set_trace()
    # ADD ANY LAYERS NECESSARY AT THE END BASED ON THE OUTPUT_LAYERS LIST
    if output_layers:
        for layer in output_layers:
            if layer =='Softmax':
                expr = relay.nn.softmax(expr)
    
    mod = relay.Module.from_expr(expr)

    return mod

'''


@relay.transform.module_pass(opt_level=4)
class ACCELModule:

    def __init__(self, path, output_layout, model_name, output_layers=None):
        self.path     = path
        self.output_layout = output_layout
        self.output_layers = output_layers
        self.model_name    = model_name
        
    def transform_module(self, mod, ctx):
        print (" passed parameters %s, %s, %s" %(self.path,self.output_layout,self.output_layers))


        mod = self.reconst_graph(mod,self.path,self.output_layout,self.model_name,self.output_layers)

        return mod


    def reconst_graph(self, mod, path, output_layout, model_name, output_layers=None):

        #pdb.set_trace()
        node_map={}
        xdnn_inputs = []
        compiler_json_file = path + model_name + "_compiler.json"
        with open(compiler_json_file) as json_file:
            json_graph = json.load(json_file)
        
        graph_inputs  = json_graph["inputs"]
        graph_outputs = json_graph["outputs"]

        #pdb.set_trace()
        compiler_shape_output = json_graph["network"][-1]["outputshapes"]
        xfuse_inputs=[]
        fuse_list=[]
        queue=[]
        #input_list = [n['input_name'] for n in graph_inputs]
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
    
