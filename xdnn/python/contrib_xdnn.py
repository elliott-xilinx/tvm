"""xDNN implementation of NN operations that can be accelerated"""


import tvm
import numpy as np
import pdb
from nnvm.top import registry as reg
from tvm.relay import op as op
from ctypes import *
import ctypes
import os
import warnings
import extern_accel


@reg.register_schedule("accel", level=15)
def schedule_accel(attrs,outputs,target):
    #pdb.set_trace()
    ##    print("-- debug: accel schedule")
    return tvm.create_schedule([x.op for x in outputs])


    
@reg.register_compute("accel", level=15)
def compute_accel(attrs,inputs,outputs):
    #pdb.set_trace()
    ##    print ("-- debug: accel compute")
    
    op = 'accel'
    name = 'accel0'
    attrs_dict = { k: attrs[k] for k in attrs.keys() }
    input_names = [inpt.op.name for inpt in inputs]
    in_shapes = [[int(i) for i in inpt.shape] for inpt in inputs]
    out_shapes = [[int(i) for i in outputs[0].shape]]
    
    # EXTERNAL FUNCTION TO RUN THE FUSED OPERATION
    out = tvm.extern(outputs[0].shape, inputs, lambda ins, outs: tvm.call_packed('tvm.accel.accel_fused', attrs['path'], attrs['output_layout'], attrs['model_name'], attrs['platform'], outs[0], *ins ), name=name)
    
      
    ##    print(out.shape)
    
    return out
    #return outputs
    

   
@op.register_schedule("nn.accel", level=15)
def schedule_accel(attrs,outputs,target):

    print("-- debug: accel schedule")
    return tvm.create_schedule([x.op for x in outputs])



@op.register_compute("nn.accel", level=15)
def compute_accel(attrs,inputs,outputs, target):
    #pdb.set_trace()
    print ("-- debug: accel compute")
    #pdb.set_trace()
    op = 'accel'
    name = 'accel0'
   
    
    # EXTERNAL FUNCTION TO RUN THE FUSED OPERATION
    out = tvm.extern(outputs.shape, inputs, lambda ins, outs: tvm.call_packed('tvm.accel.accel_fused', attrs.path,attrs.output_layout, attrs.model_name, outs[0], *ins ), name=name)


    # TEST MULTI OUTPUT
    #out_shapes = [tuple(tensor.shape) for tensor in outputs.fields]
    #out = tvm.accelern(out_shapes, inputs, lambda ins, outs: tvm.call_packed('tvm.accel.accel_fused', attrs.path,attrs.output_layout, outs, *ins ), name=name)
      
    ##    print(out.shape)
    
    return [out]
    #return outputs
