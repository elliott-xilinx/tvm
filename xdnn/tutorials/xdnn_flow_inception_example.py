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
#data_shape   = (1,3,224,224)
img_shape = (1,3,224,224)

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


# SELECT MODEL
#select_model( 'Caffe-GoogLeNet_bvlc_without_lrn' ) # NOT WORKING
#select_model( 'Tensorflow-SLIM-InceptionV1'      )
#select_model( 'Tensorflow-SLIM-VGG16'            )
#select_model( 'Tensorflow-SLIM-ResNet_V1_50'     )
#select_model( 'Tensorflow-SLIM-ResNet_V1_101'    )
#select_model( "Tensorflow-SLIM-VGG19"            )
#select_model( "Tensorflow-SLIM-ResNet_V2_152"    )
select_model( "MXNet-GLUON-ResNet_V1_18" ) 
#select_model( "MXNet-GLUON-ResNet_V1_50" )
#select_model( "MXNet-GLUON-VGG_13"    )

print("Framework: {}".format(framework))
print("Model path: {}".format(model_path))
print("Optional model path: {}".format(opt_model_path))
print("Shapes: {}".format(data_shapes))


from xfdnn.tools.io import load_model_from_file





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

#val_images = ["/workspace/MLsuite/examples/image_classify/sample_images/dog.jpg"]
val_images = ["/workspace/MLsuite/notebooks/imagenet-val/ILSVRC2012_val_00000002.JPEG"]


img = cv2.imread(val_images[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title(val_images[0])
#plt.show()

batch_array = np.empty(img_shape, dtype=np.float32, order='C')
img_paths = val_images

img_io_func = xfdnn_io.load_imgs_from_file(data_io, img_shape[2:4], model_name)

data = img_io_func(img_paths)
batch_array[:] = data
print(batch_array.shape)
print(batch_array[0])
np.set_printoptions(precision=4, suppress=True)




# READING MODEL USING NNVM/RELAY
frontend = 'NNVM'
if frontend == 'NNVM':
    compute_graph, params, data_layout = \
        load_model_from_file(frontend, framework)(model_path, 
                                                  data_shapes, 
                                                  opt_model_path)
    xfgraph = tvm_compiler.from_nnvm(compute_graph, params, shapes=data_shapes, 
                                     #output_op = "InceptionV1/Logits/AvgPool_0a_7x7/AvgPool",
                                     #output_op = "elemwise_add7",
                     data_layout=data_layout) #from_nnvm output_op
###elif frontend == 'Relay':
###    mod, params, data_layout = \
###        load_model_from_file(frontend, framework)(model_path, data_shapes, 
###                                                  opt_model_path)
###    xfgraph = tvm_compiler.from_relay(mod, params, 
###                                      data_layout=data_layout,
###                                      add_output_layers=add_output_layers)
xfgraph.visualize('tvm_graph.png')
    




 
# QUANTIZE MODEL
import xfdnn.tools.io as xfdnn_io
from xfdnn.tools.xfgraph.quantization import XfGraphDefaultQuantizer, XfGraphAddScalingQuantizer

calibration_directory = '/workspace/MLsuite/notebooks/calibration_directory'
img_io_func = xfdnn_io.load_imgs_from_file(data_io, img_shape[2:4], model_name)

quantizer = XfGraphDefaultQuantizer(
    xfgraph=xfgraph,
    quant_file=config["quantizecfg"], 
    data_layout='NCHW',
    data_loading_func=img_io_func,
    calibration_directory=calibration_directory,
    cal_size=15
)
quantizer.quantize()
 
xfgraph.save('xfgraph')
 
# COMPILE
tvm_compiler.compile(xfgraph)




# BUILD FOR CPU
from xfdnn.tools.xfgraph.xfgraph import XfGraph
xfgraph.build(device='cpu')





# SIM
inputs = {}
# TODO only one input so this is working
inputs[list(data_shapes.keys())[0]] = batch_array # Placeholder / data / 0





# RUN ON CPU FOR TESTING PURPOSES
res = xfgraph.run(inputs, #['InceptionV1/Logits/SpatialSqueeze'],
                  #['global_avg_pool2d0'],
                  #['data'],
                  batch_size=1)

print(res[0].shape)
print(repr(res[0]))
print(np.max(res[0]))






# RECONSTRUCT AND FUSE THE GRAPH FOR XDNN
import contrib_xdnn
from graph import graph_reconst
gidx = compute_graph.index
print("--debug: start reconstructing the graph")
graph = graph_reconst(json_path = config["netcfg"], nnvm_graph = gidx.nodes, output_layout = data_layout, output_layers = add_output_layers )

print("--debug: finished reconstructing the graph")





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


print("--debug: finished recompiling NNVM graph")






# RUN THE GRAPH
# TODO PROPER DATA LAYOUT/SHAPE HAS TO BE PASSED...
# IF INPUT TENSOR IS NOT THE IMAGE
from tvm.contrib import graph_runtime
ctx = tvm.cpu(0)
m = graph_runtime.create(graph, lib, ctx)
# WHEN RUNNING TVM-ONLY, NP.RESHAPE HAS TO BE CHANGED TO NP.TRANSPOSE FOR MODELS THAT DO NOT FOLLOW 'NCHW' LAYOUT SINCE THE IMAGE IS READ IN 'NCHW' LAYOUT
#m.set_input(Placeholder=np.array(res[0]))
#m.set_input(Placeholder=(np.transpose(batch_array,(0,2,3,1))))
params.update({input_name:np.reshape(np.array(batch_array),shape_dict[input_name])})
m.set_input(**params)




# RUN
m.run()
tvm_output = m.get_output(0)




def softmax(x):
    return np.exp(x - np.max(x, axis=1, keepdims=True)) / np.expand_dims(np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1), axis=1)

if frontend == 'NNVM' and (framework == 'Tensorflow' \
    and model_name in ['vgg_16', 'vgg_19']) or \
    (framework == 'MXNet' \
    and model_name in ['resnet_v1_18', 'resnet_v1_34', 'resnet_v1_50']):
    res[0] = softmax(res[0])



    


# PERFORM PREDICTION
import xfdnn.tools.xfgraph.classification as xfdnn_classification
# TODO: Make this more automatic: 1000 <-> 1001
def predict(tensor):
    raw_predictions = tensor
    if raw_predictions.shape[1] == 1000:
        label_lst = [elem[1] for elem in imagenet_val_set[:raw_predictions.shape[0]]]
        synset_words = 'synset_words.txt'
    elif raw_predictions.shape[1] == 1001:
        # for inception, ...
        label_lst = [int(elem[1]) + 1 for elem in imagenet_val_set[:raw_predictions.shape[0]]]
        synset_words = 'synset_words_1001.txt'
    else:
        raise ValueError("Unknown number of predicted categories: {}".format(raw_predictions.shape[1]))
    
    top_1 = xfdnn_classification.get_top_k_accuracy(raw_predictions, synset_words, 1, label_lst)
    top_5 = xfdnn_classification.get_top_k_accuracy(raw_predictions, synset_words, 5, label_lst)   
    print("Top 1: {}".format(top_1))
    print("Top 5: {}".format(top_5))

predict(res[0])
predict(tvm_output.asnumpy())
