import json
import pdb
import nnvm
import nnvm.symbol as sym
import os

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
    


def graph_reconst(json_path, nnvm_graph, output_layers=None): 
    node_map={}
    xdnn_inputs = []

    with open(json_path) as json_file:
        json_graph = json.load(json_file)
        
    graph_inputs = json_graph["inputs"]
    graph_outputs = json_graph["outputs"]
    graph_outputs = json_graph["outputs"]
    #pdb.set_trace()
    compiler_shape_output = json_graph["network"][-1]["outputshapes"]
    output_shape = (1,compiler_shape_output[2],compiler_shape_output[3],compiler_shape_output[1])   
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
                    # create xdnn node
                    # THE JSON_GRAPH MAY NEED TO BE CHANGED TO JSON_PATH
                    # SINCE THE PATH IS NEEDED FOR ACCESSING OTHER XDNN FILES
                    # ALSO, OS.GETCWD() MAY NEED TO BE CHANGED TO A SPECIFIC ADDRESS
                    new_entry = sym.xdnn(*xdnn_inputs, json_graph=os.getcwd(), output_shape=output_shape)
                    node_map[nid] = new_entry
        else:
                    
            if op_name == "null": #and nid != 540:
                new_entry = nnvm.symbol.Variable(node_name)
                xdnn_inputs.append(new_entry)
            else:
                children = [node_map[e[0]] for e in node["inputs"]]
                new_entry = get_clone(children, op_name, node_name, attrs)
                #continue
     
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

    
