import os
import numpy as np

import logging
logging.basicConfig()
logger = logging.getLogger('xfgraph')
# logger.setLevel(logging.DEBUG)

import xfgraph
import models.tools as model_tools

import tvm
from tvm import contrib
import nnvm
import tvm.relay as relay
from tvm.contrib import vai

##################################################
# MODEL SETTINGS
##################################################

models = model_tools.get_models_dict()

def select_model(MODEL):
    global framework, model_name, model_path, opt_model_path, data_io,\
        input_formats, data_inputs, data_shapes, add_output_layers
    
    print(models[MODEL])
    framework         = models[MODEL]['framework']
    model_name        = models[MODEL]['model']
    model_path        = models[MODEL]['model_path']
    opt_model_path    = models[MODEL]['weights_path']
    model_io          = models[MODEL]['io']
    add_output_layers = models[MODEL]['add_output_layers']
    
    input_formats     = models[MODEL]['input_formats']
    data_inputs       = models[MODEL]['inputs']
    data_input_shapes = models[MODEL]['input_shapes']
    
    data_io = {}
    data_shapes = {}
    for inpt, shape, io in zip(data_inputs, data_input_shapes, model_io):
        data_shapes[inpt] = shape
        data_io[inpt] = io

#select_model( "Tensorflow-SLIM-ResNet_V1_50")
select_model( "MXNet-GLUON-ResNet_V1_18" ) 

print("Framework: {}".format(framework))
print("Model path: {}".format(model_path))
print("Optional model path: {}".format(opt_model_path))
print("Shapes: {}".format(data_shapes))

##################################################
# INPUTS FUNC
##################################################

img_loader = ImgLoader()
data_preprocessor = ImgProcessor(
    proc_key = data_io[data_inputs[0]]
)

def inputs_func(iter, layout):
    # Should be standalone, e.g. don't use variables from outside 
    #   the function
    FILE_PATH = os.getcwd()
    file_dir = os.path.join(FILE_PATH, "imagenet/val-small")

    img_files = [os.path.join(file_dir, f) for f in os.listdir(file_dir)]

    # img loader and processor load in NHWC format
    imgs = img_loader.load(img_files)
    data = data_preprocessor.execute(imgs)
    
    if layout == 'NCHW':
        data = np.transpose(data, (0,3,1,2))
    
    return data

##################################################
# BUILD & EXECUTE
##################################################

from xfgraph.io import load_model_from_file

target        = tvm.target.arm_cpu('ultra96')
input_name    = list(data_shapes.keys())[0]
shape_dict    = data_shapes


frontend = 'NNVM'

if frontend == 'NNVM':
    compute_graph, params, data_layout = \
        load_model_from_file(frontend, framework)\
            (model_path, data_shapes, opt_model_path)
    
    nnvm_graph = vai.NNVMPartitioningPass(target='dpu',
        params=params, data_shapes=shape_dict, 
        inputs_func=inputs_func, layout=data_layout)

    params_dtypes = dict((k, params[k].dtype) for k in params)

    input_type    = 'float32'
    dtype_dict    = {input_name: input_type }
    dtype_dict.update(params_dtypes)
    
    graph, lib, params = nnvm.compiler.build(
        graph, target, shape_dict, dtype_dict,
        params=params)

    lib.export_library("tvm_dpu_cpu.so", contrib.cc.create_shared, 
        cc="/usr/aarch64-linux-gnu/bin/ld")
    with open("tvm_dpu_cpu.json","w") as f:
        f.write(graph.json())

    with open("tvm_dpu_cpu.params", "wb") as f:
        f.write(relay.save_param_dict(params))

elif frontend == 'Relay':
    mod, params, data_layout = \
        load_model_from_file(frontend, framework)\
            (model_path, data_shapes, opt_model_path)

    mod = vai.PartitioningPass(target='dpu', params=params, 
        inputs_func=inputs_func, layout=data_layout)(mod)

    graph, lib, params = relay.build_module.build(
        mod, target, params=params)

    # SAVE NNVM/TVM OUTPUT
    lib.export_library("tvm_dpu_cpu.so", contrib.cc.create_shared, 
        cc="/usr/aarch64-linux-gnu/bin/ld")
    with open("tvm_dpu_cpu.json","w") as f:
        f.write(graph)

    with open("tvm_dpu_cpu.params", "wb") as f:
        f.write(relay.save_param_dict(params))

    
    
