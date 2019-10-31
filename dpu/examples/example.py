
import os
import numpy as np

import logging
logging.basicConfig()
logger = logging.getLogger('xfgraph')
# logger.setLevel(logging.DEBUG)

# xfgraph imports
import xfgraph
import models.tools as model_tools
# Import dnndk at the top of module because
#   this registers the dpu as a xfgraph target device
from xfgraph.contrib import dnndk

##################################################
# MODEL SETTINGS
##################################################

models = model_tools.get_models_dict()

def select_model(MODEL):
    global framework, model_name, model_path, opt_model_path, data_io,\
        input_formats, data_inputs, data_shapes, add_output_layers
    
    
    #model_name = MODEL #'TF-GoogLeNet_bvlc_without_lrn' # # #'TF-ResNet50' #
    
    print(models[MODEL])
    framework = models[MODEL]['framework']
    model_name = models[MODEL]['model']
    model_path = models[MODEL]['model_path']
    opt_model_path = models[MODEL]['weights_path']
    data_io = models[MODEL]['io']
    add_output_layers = models[MODEL]['add_output_layers']
    
    input_formats = models[MODEL]['input_formats']
    data_inputs = models[MODEL]['inputs']
    data_input_shapes = models[MODEL]['input_shapes']
    data_shapes = {}
    for inpt, shape in zip(data_inputs, data_input_shapes):
        data_shapes[inpt] = shape

select_model( "Tensorflow-SLIM-ResNet_V1_50")

print("Framework: {}".format(framework))
print("Model path: {}".format(model_path))
print("Optional model path: {}".format(opt_model_path))
print("Shapes: {}".format(data_shapes))

##################################################
# LOAD MODEL
##################################################


from xfgraph.io import load_model_from_file
from xfgraph.frontend import from_nnvm, from_relay

frontend = 'NNVM'

if frontend == 'NNVM':
    compute_graph, params, data_layout = \
        load_model_from_file(frontend, framework)(model_path, 
                                                  data_shapes, 
                                                  opt_model_path)
    xfgraph = from_nnvm(compute_graph, params, shapes=data_shapes, 
                        #output_op = "InceptionV3/InceptionV3/Conv2d_4a_3x3/Relu", #resnet_v2_50/block1/unit_1/bottleneck_v2/preact/Relu
                        data_layout=data_layout,
                        add_output_layers=add_output_layers) 
elif frontend == 'Relay':
    mod, params, data_layout = \
        load_model_from_file(frontend, framework)(model_path, data_shapes, 
                                                  opt_model_path)
    xfgraph = from_relay(mod, params, 
                         data_layout=data_layout,
                         add_output_layers=add_output_layers)

##################################################
# COMPILATION/QUANTIZATION
##################################################

# Optimization
from xfgraph.generator.tensorflow import XfGraphTfGeneratorOptimizer
xfgraph.optimize(XfGraphTfGeneratorOptimizer)

# Partitioning
xfgraph.partition(devices=['dpu'])

# Dump the partitioned graph -> dpu_xgraph.json - dpu_xgraph.h5
from xfgraph.graph.io.xgraph_io import XGraphIO

dpu_xgraph = xfgraph.schedule(device='dpu')
XGraphIO.save(dpu_xgraph, 'dpu_xgraph')

# Quantization
from xfgraph.contrib.dnndk.decent_quantizer import DECENTQuantizer

FILE_PATH = os.getcwd()
cal_dir = os.path.join(FILE_PATH, "imagenet/val-small")

quantizer = DECENTQuantizer(
    xfgraph = xfgraph,
    data_prep_key = data_io,
    cal_dir = cal_dir,
    cal_size=32,
    cal_iter=4
)

# Run: this creates the work/deploy_model.pb and work/quantize_eval_model.pb
netcfgs = quantizer.quantize(partitioning=True)

# Compilation
from xfgraph.contrib.dnndk.dnnc_compiler import DNNCCompiler

FILE_PATH = os.getcwd()
output_dir = os.path.join(FILE_PATH, "work")
netcfgs = {
    'xp0': os.path.join(FILE_PATH, "work/deploy_model.pb")
}
dcf = "/dnndk/dcf/ZCU104.dcf"

compiler = DNNCCompiler(
    xfgraph = xfgraph,
    netcfgs = netcfgs,
    dcf = dcf
)

# Run: This creates the DPU library file containing the full DPU network and params
#  libdpumodelxp0.so and a compatibility json file for input/output naming
compiler.compile()



