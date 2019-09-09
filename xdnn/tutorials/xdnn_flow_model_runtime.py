import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

import os

import tvm
import nnvm
import tvm.relay as relay

import numpy as np

import os

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
val_images = ["/workspace/MLsuite/examples/image_classify/sample_images/dog.jpg"]
imagenet_val_list = [('dog.jpg',259)]

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
# TVM RUN
##################################################
import contrib_xdnn

tvm_name = "tvm_cpu"
#tvm_name = "tvm_fpga_cpu"

# Load the module, graph and params.
print(f"Loading model artifact: {tvm_name}")
loaded_lib = tvm.module.load(f"{tvm_name}/{tvm_name}.so")
loaded_json = open(f"{tvm_name}/{tvm_name}.json").read() 
loaded_params = bytearray(open(f"{tvm_name}/{tvm_name}.params", "rb").read())
 
from tvm.contrib import graph_runtime
m = graph_runtime.create(loaded_json, loaded_lib, tvm.cpu(0))
 
# Initialize the parameters.
params = nnvm.compiler.load_param_dict(loaded_params)
m.load_params(loaded_params)
 
# RUN
input_name    = list(data_shapes.keys())[0]
shape_dict    = data_shapes

params.update({input_name:np.reshape(np.array(batch_array),shape_dict[input_name])})
m.set_input(**params)
 
# Execute the Model 
m.run()
 
# Get the first output
 
tvm_output = m.get_output(0)
 
##################################################
# Final output
##################################################
tvm_output = tvm_output.asnumpy()
if tvm_name == "tvm_cpu":
    for op in add_output_layers:
        if op == "Softmax":
            tvm_output = softmax(tvm_output)
        else:
            print("Output layer \"{}\" yet supported.".format(op))

predict("TVM STANDALONE",tvm_output, imagenet_val_labels)
