import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

#from ipywidgets import interact

import tvm
import nnvm
import nnvm.symbol as sb
import tvm.relay as relay

import numpy as np

import os
import shutil


from xfdnn.tools.compile.bin.xfdnn_compiler_tvm import TVMCompiler

##################################################
# MESSAGE SETTINGS
##################################################
# DEBUG
#import messages
#messages.DEBUG(True)
#import pdb

# Enable for XDNN Runtime info...
#os.environ['XDNN_VERBOSE'] = "1"

np.set_printoptions(precision=4, suppress=True)

##################################################
# MODEL SETTINGS
##################################################

import models_util.model_util as model_util
models = model_util.get_models_dict()

#@interact(MODEL=sorted(models.keys()))
def select_model(MODEL):
    global framework, model_name, model_path, opt_model_path, data_io,\
        data_shapes, data_formats, add_output_layers
    
    print(models[MODEL])
    framework         = models[MODEL]['framework']
    model_name        = models[MODEL]['model']
    model_path        = models[MODEL]['model_path']
    opt_model_path    = models[MODEL]['weights_path']
    data_io           = models[MODEL]['io']
    add_output_layers = models[MODEL]['add_output_layers']
    
    data_inputs       = models[MODEL]['inputs']
    data_input_shapes = models[MODEL]['input_shapes']
    data_input_formats = models[MODEL]['input_formats']
    data_shapes = {}
    data_formats = {}
    for inpt, shape in zip(data_inputs, data_input_shapes):
        data_shapes[inpt] = shape
    for inpt, inpt_format in zip(data_inputs, data_input_formats):
        data_formats[inpt] = inpt_format


# SELECT MODEL
#select_model( 'Caffe-GoogLeNet_bvlc_without_lrn' ) # NOT WORKING
###select_model( 'Tensorflow-SLIM-InceptionV1'      )
#select_model( 'Tensorflow-SLIM-VGG16'            )
#select_model( 'Tensorflow-SLIM-ResNet_V1_50'     )
#select_model( 'Tensorflow-SLIM-ResNet_V1_101'    )
#select_model( "Tensorflow-SLIM-VGG19"            )
#select_model( "Tensorflow-SLIM-ResNet_V2_152"    )
#select_model( "MXNet-GLUON-ResNet_V1_18" ) 
select_model( "MXNet-GLUON-ResNet_V1_50" )
#select_model( "MXNet-GLUON-VGG_13"    )

print("Framework: {}".format(framework))
print("Model path: {}".format(model_path))
print("Optional model path: {}".format(opt_model_path))
print("Shapes: {}".format(data_shapes))



##################################################
# DATASET/VALIDATION SETUP
##################################################

# PREPARING THE INPUT
# CHOSE AN IMAGE TO RUN, DISPLAY IT FOR REFERENCE
import xfdnn.tools.io as xfdnn_io
import numpy as np
import cv2

from matplotlib import pyplot as plt
#%matplotlib inline

with open('/workspace/MLsuite/notebooks/imagenet-val/val_map.txt') as f:
    imagenet_val_list = [line.strip('\n').split(' ') for line in f.readlines()]

# TODO: Update reshape to allow variable batch size...
val_images = ["/workspace/MLsuite/notebooks/imagenet-val/ILSVRC2012_val_00000001.JPEG"]

## Non-ImageNet...
#val_images = ["/workspace/MLsuite/examples/image_classify/sample_images/dog.jpg"]
#imagenet_val_list = [('dog.jpg',259)]

imagenet_val_dict = dict(imagenet_val_list)

# Create image_val_labels to match with val_images
imagenet_val_labels = [(os.path.basename(fname),imagenet_val_dict[os.path.basename(fname)]) for fname in val_images]

debug_image = False
if debug_image:
    img = cv2.imread(val_images[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(val_images[0])
    plt.show()

img_paths = val_images

# Uses model name because some models require special combinations of cropping, resize, channel swap, etc.
# Assume only one input for these test classification models.  Potentially 0 is not the same...
shape_list = list(data_shapes.values())
format_list = list(data_formats.values())

resize_shape = shape_list[0][2:4] if format_list[0] == "NCHW" else shape_list[0][1:3]

img_io_func = xfdnn_io.load_imgs_from_file(data_io, resize_shape, model_name)

data = img_io_func(img_paths)
batch_array = np.empty(data.shape, dtype=np.float32, order='C')
batch_array[:] = data


# Utilities for accuracy report
import xfdnn.tools.xfgraph.classification as class_util

# TODO: Make this more automatic: 1000 <-> 1001
def predict(desc,tensor, val_labels):
    #import pdb; pdb.set_trace()
    print("\n=========================\nPrediction: {}".format(desc))
    raw_predictions = tensor
    if raw_predictions.shape[1] == 1000:
        label_lst = [elem[1] for elem in val_labels[:raw_predictions.shape[0]]]
        synset_words = 'synset_words.txt'
    elif raw_predictions.shape[1] == 1001:
        # for inception, ...
        label_lst = [int(elem[1]) + 1 for elem in val_labels[:raw_predictions.shape[0]]]
        synset_words = 'synset_words_1001.txt'
    else:
        raise ValueError("Unknown number of predicted categories: {}".format(raw_predictions.shape[1]))
    
    top_1 = class_util.get_top_k_accuracy(raw_predictions, synset_words, 1, label_lst)
    top_5 = class_util.get_top_k_accuracy(raw_predictions, synset_words, 5, label_lst)   
    print("Top 1: {}".format(top_1))
    print("Top 5: {}".format(top_5))
    print("=========================")

def softmax(x):
    return np.exp(x - np.max(x, axis=1, keepdims=True)) / np.expand_dims(np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1), axis=1)


##################################################
# TVM COMPILATION/PARTITIONING
##################################################
model_dir = "tvm_fpga_cpu"
try:
    if False and os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
except:
    pass
    
config = {
    'netcfg': model_dir+"/tvm_compiler.json",
    'weights': model_dir+"/weights_data.h5",
    'quantizecfg': model_dir+"/tvm_quantizer.json"
}

# Initializes TVM Compiler for XDNN
xdnn_tvm_compiler = TVMCompiler(
    netcfg=config['netcfg'],
    weights=config['weights']
)

# Wrapper for reading models into NNVM/RELAY format
from xfdnn.tools.io import load_model_from_file

# READING MODEL USING NNVM/RELAY
frontend = 'NNVM'
if frontend == 'NNVM':
    
    compute_graph, params, data_layout = \
        load_model_from_file(frontend, framework)(model_path, 
                                                  data_shapes, 
                                                  opt_model_path)
    
    xfgraph = xdnn_tvm_compiler.from_nnvm(compute_graph, params, shapes=data_shapes, 
                                     #output_op = "InceptionV1/Logits/AvgPool_0a_7x7/AvgPool",
                                     #output_op = "elemwise_add7",
                                     data_layout=data_layout) #from_nnvm output_op
###elif frontend == 'Relay':
###    mod, params, data_layout = \
###        load_model_from_file(frontend, framework)(model_path, data_shapes, 
###                                                  opt_model_path)
###    xfgraph = xdnn_tvm_compiler.from_relay(mod, params, 
###                                      data_layout=data_layout,
###                                      add_output_layers=add_output_layers)

# xfgraph is basically our XDNN IR
xfgraph.visualize('tvm_graph.png')

################################################## 
# XDNN QUANTIZE MODEL
################################################## 

import xfdnn.tools.io as xfdnn_io
from xfdnn.tools.xfgraph.quantization import XfGraphDefaultQuantizer, XfGraphAddScalingQuantizer

calibration_directory = '/workspace/MLsuite/notebooks/calibration_directory'
img_io_func = xfdnn_io.load_imgs_from_file(data_io, resize_shape, model_name)

"""
quantizer = XfGraphDefaultQuantizer(
    xfgraph=xfgraph,
    quant_file=config["quantizecfg"], 
    data_layout='NCHW',
    data_loading_func=img_io_func,
    calibration_directory=calibration_directory,
    cal_size=15
)
quantizer.quantize()
"""

xfgraph.save('xfgraph')


# COMPILE
xdnn_tvm_compiler.compile(xfgraph)

# BUILD FOR CPU
from xfdnn.tools.xfgraph.xfgraph import XfGraph
xfgraph.build(device='cpu')





# SIM
inputs = {}
# TODO only one input so this is working
inputs[list(data_shapes.keys())[0]] = batch_array # Placeholder / data / 0


##################################################
# CPU NON-TVM RUN
##################################################

# RUN ON CPU FOR TESTING PURPOSES
cpu_nontvm_res = xfgraph.run(inputs,
                      #['InceptionV1/Logits/SpatialSqueeze'],
                      #['global_avg_pool2d0'],
                      #['data'],
                      batch_size=1)

#print("Max Score:",np.max(cpu_res[0]))
#print("Max Category ID:",np.argmax(cpu_res[0]))
#predict("CPU ONLY (NON-TVM)", cpu_nontvm_res[0], imagenet_val_labels)



##################################################
# CPU TVM RUN
##################################################

# SETUP AND COMPILE THE RECONSTRUCTED NNVM GRAPH
target, target_host = 'llvm', 'llvm'
#params_dtypes = dict((k, params[k].dtype) for k in params)
input_name    = list(data_shapes.keys())[0]
shape_dict    = data_shapes
input_type    = 'float32'
dtype_dict    = {input_name: input_type }
##dtype_dict.update(params_dtypes)

graph, lib, params = nnvm.compiler.build(
    compute_graph, target, shape_dict, dtype_dict,
    params=params, target_host=target_host)

# RUN THE GRAPH
# TODO PROPER DATA LAYOUT/SHAPE HAS TO BE PASSED...
# IF INPUT TENSOR IS NOT THE IMAGE
from tvm.contrib import graph_runtime
ctx = tvm.cpu(0)
m = graph_runtime.create(graph, lib, ctx)

# WHEN RUNNING TVM-ONLY, NP.RESHAPE HAS TO BE CHANGED TO NP.TRANSPOSE FOR MODELS THAT DO NOT FOLLOW 'NCHW' LAYOUT SINCE THE IMAGE IS READ IN 'NCHW' LAYOUT
#m.set_input(Placeholder=np.array(res[0]))
#m.set_input(Placeholder=(np.transpose(batch_array,(0,2,3,1))))

# SAVE NNVM/TVM OUTPUT
model_dir = "tvm_cpu"
try:
    if False and os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
except:
    pass
lib.export_library(model_dir+"/tvm_cpu.so")
with open(model_dir+"/tvm_cpu.json","w") as f:
    f.write(graph.json())
with open(model_dir+"/tvm_cpu.params","wb") as f:
    f.write(nnvm.compiler.save_param_dict(params))

# RUN
params.update({input_name:np.reshape(np.array(batch_array),shape_dict[input_name])})
m.set_input(**params)

m.run()
cpu_tvm_output = m.get_output(0)


##################################################
# FPGA TVM RUN
##################################################

# RECONSTRUCT AND FUSE THE GRAPH FOR XDNN
import contrib_xdnn
from graph import graph_reconst
gidx = compute_graph.index

print("Starting to partitioning/reconstructing the graph")
graph = graph_reconst(json_path = config["netcfg"], nnvm_graph = gidx.nodes, output_layout = data_layout, output_layers = add_output_layers )
print("Finished partitioning/reconstructing the graph")

# SETUP AND COMPILE THE RECONSTRUCTED NNVM GRAPH
target, target_host = 'llvm', 'llvm'
params_shapes = dict((k, params[k].shape) for k in params)
params_dtypes = dict((k, params[k].dtype) for k in params)
input_name    = list(data_shapes.keys())[0]
shape_dict    = data_shapes
input_type    = 'float32'
dtype_dict    = {input_name: input_type }
dtype_dict.update(params_dtypes)

graph, lib, params = nnvm.compiler.build(
    graph, target, shape_dict, dtype_dict,
    params=params, target_host=target_host)

print("Finished recompiling NNVM/TVM graph")


# RUN THE GRAPH
# TODO PROPER DATA LAYOUT/SHAPE HAS TO BE PASSED...
# IF INPUT TENSOR IS NOT THE IMAGE
from tvm.contrib import graph_runtime
ctx = tvm.cpu(0)
m = graph_runtime.create(graph, lib, ctx)

# WHEN RUNNING TVM-ONLY, NP.RESHAPE HAS TO BE CHANGED TO NP.TRANSPOSE FOR MODELS THAT DO NOT FOLLOW 'NCHW' LAYOUT SINCE THE IMAGE IS READ IN 'NCHW' LAYOUT
#m.set_input(Placeholder=np.array(res[0]))
#m.set_input(Placeholder=(np.transpose(batch_array,(0,2,3,1))))


# SAVE NNVM/TVM OUTPUT
model_dir = "tvm_fpga_cpu"
lib.export_library(model_dir+"/tvm_fpga_cpu.so")
with open(model_dir+"/tvm_fpga_cpu.json","w") as f:
    f.write(graph.json())
with open(model_dir+"/tvm_fpga_cpu.params","wb") as f:
    f.write(nnvm.compiler.save_param_dict(params))

            
# RUN            
params.update({input_name:np.reshape(np.array(batch_array),shape_dict[input_name])})
m.set_input(**params)

m.run()
fpga_tvm_output = m.get_output(0)

##################################################
# Final output
##################################################

for op in add_output_layers:
    if op == "Softmax":
        # Why only on cpu_res???
        cpu_nontvm_res[0] = softmax(cpu_nontvm_res[0])
        cpu_tvm_output = softmax(cpu_tvm_output.asnumpy())
    else:
        print("Output layer \"{}\" yet supported.".format(op))

predict("CPU ONLY (NON-TVM)", cpu_nontvm_res[0], imagenet_val_labels)
predict("TVM CPU ONLY",cpu_tvm_output, imagenet_val_labels)
predict("TVM CPU+FPGA",fpga_tvm_output.asnumpy(), imagenet_val_labels)
