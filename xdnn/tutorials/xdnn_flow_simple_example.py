import os
import nnvm
import nnvm.symbol as sym
import tvm
import numpy as np
import tensorflow as tf

from xfdnn.tools.compile.bin.xfdnn_compiler_tvm import TVMCompiler
import pdb

import messages
import contrib_xdnn
from graph import graph_reconst
from tvm.contrib import graph_runtime
#import json

messages.DEBUG(True)

os.environ['XDNN_VERBOSE'] = "1"
# TVM compiler and quantizer

config = {
    'netcfg': "work/tvm_compiler.json",
    'weights': "work/weights_data.h5",
    'quantizecfg': "work/tvm_quantizer.json"
}

nnvm_compiler = TVMCompiler(
    netcfg=config['netcfg'],
    weights=config['weights']
)

#pdb.set_trace()
# Example 3: Convolution and max pool
    # Example 3: Convolution and max pool
input_shape = (1,1,4,4)
output_shape = (1,2,2,2)
input_names = ['x']

data = np.reshape(np.array([[10,10,0,40],[50,10,0,80],[30,50,10,0],[10,90,30,40]]), input_shape)
weight = np.reshape(np.array([[[1,2],[3,0]],[[1,1],[0,1]]], dtype=np.float32), (2,1,2,2))


x = sym.Variable("x")
y = sym.conv2d(x, channels = 2, kernel_size=[2,2], use_bias=False)
z = sym.max_pool2d(y, pool_size=[2,2], strides=[1,1])
#h = sym.__add_scalar__(z,scalar= np.double(15.0))
compute_graph = nnvm.graph.create(z)


#pdb.set_trace()
#TEMP


xfgraph = \
          nnvm_compiler.from_nnvm(compute_graph, 
                                 params={ 'conv2d0_weight': weight}, 
                                 shapes={'x': input_shape}, 
                                 data_layout='NCHW') 



## QUANTIZER
import xfdnn.tools.io as xfdnn_io

quant_inputs=data
#calibration_directory = '/workspace/MLsuite/notebooks/calibration_directory'
#data_io = 'Caffe-format'
#img_io_func = xfdnn_io.load_imgs_from_file(data_io, input_shape[2:4], 'model_name')

xfgraph.quantize(config["quantizecfg"], 
                 force_inputs=quant_inputs,
                 data_layout='NCHW')


nnvm_compiler.compile(xfgraph)


## CPU
xfgraph.build(device='cpu')
inputs = { 'x': data }
cpu_out = xfgraph.run(inputs)
print(cpu_out)


## QUANTIZATION SIMULATOR
xfgraph.build(device='sim', quantcfg=config["quantizecfg"])
inputs = { 'x': data }
sim_out = xfgraph.run(inputs) #outputs=['x_quantize'])
print(sim_out)

# FPGA 
# Make sure you have access to a machine with FPGA installed
#pdb.set_trace()
#xfgraph.build(device='fpga', 
#              quantcfg=config["quantizecfg"], 
#              fpga_netcfg=config["netcfg"], 
#              fpga_params_file=config["weights"]) #



target, target_host = 'llvm', 'llvm'
params={}
shape_dict = { 'x': input_shape}
input_type = 'float32'
dtype_dict = { 'x': input_type}
 
ctx = tvm.cpu(0)
config['quantizecfg']=""
config['weights']=""

# RECONSTRUCT AND FUSE THE GRAPH FOR XDNN
gidx = compute_graph.index
graph = graph_reconst(config["netcfg"],gidx.nodes)


# COMPILE THE RECONSTRUCTED NNVM GRAPH
graph, lib, params = nnvm.compiler.build(
    graph, target, shape_dict, dtype_dict,
    params=params, target_host=target_host)

#pdb.set_trace()

# RUN THE GRAPH
m = graph_runtime.create(graph, lib, ctx)
m.set_input(x=data)
# RUN
m.run()
out = tvm.nd.empty((output_shape), 'float32')
tvm_output = m.get_output(0,out)
print(tvm_output)
