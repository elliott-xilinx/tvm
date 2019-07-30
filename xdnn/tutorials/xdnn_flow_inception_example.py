import os
from ipywidgets import interact

import tvm
import nnvm
import nnvm.symbol as sb
import tvm.relay as relay

import numpy as np
import tensorflow as tf

# DEBUG
import messages
messages.DEBUG(True)
import pdb

from xfdnn.tools.compile.bin.xfdnn_compiler_tvm import TVMCompiler



#from graph import graph_reconst

os.environ['XDNN_VERBOSE'] = "1"

# DATA
data_shape   = (1,3,224,224)
output_shape = (1,2,2,2)

# TVM compiler

config = {
    'netcfg': "work/tvm_compiler.json",
    'weights': "work/weights_data.h5",
    'quantizecfg': "work/tvm_quantizer.json"
}

tvm_compiler = TVMCompiler(
    netcfg=config['netcfg'],
    weights=config['weights']
)

import models_util.model_util as model_util
models = model_util.get_models_dict()

#@interact(MODEL=sorted(models.keys()))
def select_model(MODEL):
    global framework, model_name, model_path, opt_model_path, data_io,\
        data_shapes, add_output_layers
    
    
    #model_name = MODEL #'TF-GoogLeNet_bvlc_without_lrn' # # #'TF-ResNet50' #
    
    print(models[MODEL])
    framework         = models[MODEL]['framework']
    model_name        = models[MODEL]['model']
    model_path        = models[MODEL]['model_path']
    opt_model_path    = models[MODEL]['weights_path']
    data_io           = models[MODEL]['io']
    add_output_layers = models[MODEL]['add_output_layers']
    
    data_inputs       = models[MODEL]['inputs']
    data_input_shapes = models[MODEL]['input_shapes']
    data_shapes = {}
    for inpt, shape in zip(data_inputs, data_input_shapes):
        data_shapes[inpt] = shape

#select_model('Caffe-GoogLeNet_bvlc_without_lrn')
select_model('Tensorflow-SLIM-InceptionV1')
        
print("Framework: {}".format(framework))
print("Model path: {}".format(model_path))
print("Optional model path: {}".format(opt_model_path))
print("Shapes: {}".format(data_shapes))


from xfdnn.tools.io import load_model_from_file

frontend = 'NNVM'


#data_layout = 'NHWC'
#filter_layout = 'HWIO'

if frontend == 'NNVM':
    compute_graph, params, data_layout = \
        load_model_from_file(frontend, framework)(model_path, 
                                                  data_shapes, 
                                                  opt_model_path)
    xfgraph = tvm_compiler.from_nnvm(compute_graph, params, shapes={}, 
                                     output_op = "InceptionV1/Logits/AvgPool_0a_7x7/AvgPool",
                                     #output_op = "InceptionV1/Logits/Conv2d_0c_1x1/Conv2D",
                     data_layout=data_layout) #from_nnvm output_op
###elif frontend == 'Relay':
###    mod, params, data_layout = \
###        load_model_from_file(frontend, framework)(model_path, data_shapes, 
###                                                  opt_model_path)
###    xfgraph = tvm_compiler.from_relay(mod, params, 
###                                      data_layout=data_layout,
###                                      add_output_layers=add_output_layers)

    

###pdb.set_trace()
###xfgraph.visualize('tvm_graph.png')

# QUANTIZE

import xfdnn.tools.io as xfdnn_io
from xfdnn.tools.xfgraph.quantization import XfGraphAddScalingQuantizer
calibration_directory = '/workspace/MLsuite/notebooks/calibration_directory'
img_io_func = xfdnn_io.load_imgs_from_file(data_io, data_shape[2:4], model_name)
 
###xfgraph.quantize(config["quantizecfg"], 
###                 data_layout='NCHW',
###                 data_loading_func=img_io_func,
###                 calibration_directory=calibration_directory)
###                 #quantization_class=XfGraphAddScalingQuantizer) #, stop='resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/Conv2D')
###xfgraph.save('xfgraph')

#pdb.set_trace()
# COMPILE
tvm_compiler.compile(xfgraph)




# BUILD FOR CPU
from xfdnn.tools.xfgraph.xfgraph import XfGraph
##xfgraph = XfGraph()
##xfgraph.load('xfgraph.json', 'xfgraph.h5')
## 
#xfgraph.build(device='sim', quantcfg=config["quantizecfg"])
xfgraph.build(device='cpu')

# PREPARING THE INPUT
# CHOSE AN IMAGE TO RUN, DISPLAY IT FOR REFERENCE
import xfdnn.tools.io as xfdnn_io
import numpy as np
import cv2

from matplotlib import pyplot as plt
#%matplotlib inline

imagenet_val_set = None
with open('/workspace/MLsuite/notebooks/imagenet-val/val_map.txt') as f:
    imagenet_val_set = [line.strip('\n').split(' ') for line in f.readlines()]

# NEXT TWO VARIABLES NEED TO BE ADJUSTED TO TRY OUT OTHER INPUTS
val_images = ["/workspace/MLsuite/examples/image_classify/sample_images/dog.jpg"]
input_shape = (1,3,224,224) 

img = cv2.imread(val_images[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title(val_images[0])
#plt.show()

batch_array = np.empty(input_shape, dtype=np.float32, order='C')
img_paths = val_images

img_io_func = xfdnn_io.load_imgs_from_file(data_io, input_shape[2:4], model_name)

data = img_io_func(img_paths)
batch_array[:] = data
print(batch_array.shape)
print(batch_array[0])


np.set_printoptions(precision=4, suppress=True)

# SIM
inputs = {}
# TODO only one input so this is working
inputs[list(data_shapes.keys())[0]] = batch_array # Placeholder / data / 0


# RUN ON CPU
res = xfgraph.run(inputs, ['InceptionV1/Logits/AvgPool_0a_7x7/AvgPool'],
                  batch_size=1)

print(res[0].shape)
print(repr(res[0]))
print(np.max(res[0]))



# RECONSTRUCT AND FUSE THE GRAPH FOR XDNN
import contrib_xdnn
from graph import graph_reconst
gidx = compute_graph.index
print("--debug: start reconstructing the graph")
graph = graph_reconst(config["netcfg"],gidx.nodes)

print("--debug: finished reconstructing the graph")


#asd = nnvm.compiler.graph_util.infer_shape(compute_graph)

# COMPILE THE RECONSTRUCTED NNVM GRAPH
#pdb.set_trace()
target, target_host = 'llvm', 'llvm'
params={}
shape_dict = { 'Placeholder': input_shape, 'InceptionV1/Logits/Conv2d_0c_1x1/biases':(1001,) }
input_type = 'float32'
dtype_dict = { 'Placeholder': input_type, 'InceptionV1/Logits/Conv2d_0c_1x1/biases': input_type}
#pdb.set_trace()
graph, lib, params = nnvm.compiler.build(
    graph, target, shape_dict, dtype_dict,
    params=params, target_host=target_host)

print("--debug: finished recompiling NNVM graph")

#pdb.set_trace()

# RUN THE GRAPH
from tvm.contrib import graph_runtime
ctx = tvm.cpu(0)
m = graph_runtime.create(graph, lib, ctx)
m.set_input(Placeholder=np.array(batch_array))
# RUN
m.run()
#out = tvm.nd.empty((1,1,1001,1), 'float32')
out = tvm.nd.empty((1,1,1,1024), 'float32')
#pdb.set_trace()
tvm_output = m.get_output(0,out)
print ("--debug: EXPECTED")
print(repr(res[0]))
print ("--debug: ACTUAL")
print(np.reshape(tvm_output.asnumpy(),(1,1024,1,1)))

