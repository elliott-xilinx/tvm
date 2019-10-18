"""xDNN implementation of NN operations that can be accelerated"""


import tvm
import numpy as np
import pdb
from nnvm.top import registry as reg
from tvm.relay import op as op
from ctypes import *
import ctypes
import os

import xfdnn.rt.xdnn as xdnn
import xfdnn.rt.xdnn_io as xdnn_io
#from xfdnn.rt import xdnn, xdnn_io

@reg.register_schedule("ext", level=15)
def schedule_ext(attrs,outputs,target):
    #pdb.set_trace()
    ##    print("-- debug: ext schedule")
    return tvm.create_schedule([x.op for x in outputs])


    
@reg.register_compute("ext", level=15)
def compute_ext(attrs,inputs,outputs):
    #pdb.set_trace()
    ##    print ("-- debug: ext compute")
    
    op = 'ext'
    name = 'ext0'
    attrs_dict = { k: attrs[k] for k in attrs.keys() }
    input_names = [inpt.op.name for inpt in inputs]
    in_shapes = [[int(i) for i in inpt.shape] for inpt in inputs]
    out_shapes = [[int(i) for i in outputs[0].shape]]
    
    # EXTERNAL FUNCTION TO RUN THE FUSED OPERATION
    out = tvm.extern(outputs[0].shape, inputs, lambda ins, outs: tvm.call_packed('tvm.ext.ext_fused', attrs['path'], attrs['output_layout'], attrs['model_name'], outs[0], *ins ), name=name)
    
      
    ##    print(out.shape)
    
    return out
    #return outputs
    

   
@op.register_schedule("nn.ext", level=15)
def schedule_ext(attrs,outputs,target):

    print("-- debug: ext schedule")
    return tvm.create_schedule([x.op for x in outputs])



@op.register_compute("nn.ext", level=15)
def compute_ext(attrs,inputs,outputs, target):
    #pdb.set_trace()
    print ("-- debug: ext compute")
    #pdb.set_trace()
    op = 'ext'
    name = 'ext0'
    #attrs_dict = { k: attrs[k] for k in attrs.keys() }
    #input_names = [inpt.op.name for inpt in inputs]
    #in_shapes = [[int(i) for i in inpt.shape] for inpt in inputs]


    

    
    # EXTERNAL FUNCTION TO RUN THE FUSED OPERATION
    out = tvm.extern(outputs.shape, inputs, lambda ins, outs: tvm.call_packed('tvm.ext.ext_fused', attrs.path,attrs.output_layout, attrs.model_name, outs[0], *ins ), name=name)


    # TEST MULTI OUTPUT
    #out_shapes = [tuple(tensor.shape) for tensor in outputs.fields]
    #out = tvm.extern(out_shapes, inputs, lambda ins, outs: tvm.call_packed('tvm.ext.ext_fused', attrs.path,attrs.output_layout, outs, *ins ), name=name)
      
    ##    print(out.shape)
    
    return [out]
    #return outputs
    
    
# TODO: ADD MODEL NAME FOR NNVM
@tvm.register_func("tvm.ext.ext_fused")
def ext_fused(graph_path, output_layout, model_name, out, *ins ):

    #pdb.set_trace()
    # TEMP
    ######path   = c_char_p(graph_path.value).value
    ######layout = c_char_p(output_layout.value).value


      
   
    # CREATE A HANDLE FOR FPGA COMMUNICATION
    platform = os.environ.get('MLSUITE_PLATFORM')#"alveo-u200"
    xclbin = "/workspace/MLsuite/overlaybins/" + platform + "/overlay_4.xclbin"
    
    #if (isinstance(graph_path,str)):
    path   = graph_path    #c_char_p(graph_path.value).value
    layout = output_layout #c_char_p(output_layout.value).value
    
    args_dict = {
        'xclbin'     : xclbin,
        'netcfg'     : str(path + model_name + "_compiler.json" ),
        'quantizecfg': str(path + model_name + "_quantizer.json"),
        'weights'    : str(path + model_name + "_weights.h5"    ), 
        'scaleA'     : 1,
        'scaleB'     : 1,
        'PE'         : 0,
        'batch_sz'   : ins[0].shape[0],
        'inshape'    : tuple(ins[0].shape[1:])
    }

          
    ######else:
    ######    path       = c_char_p(graph_path.value).value
    ######    layout     = c_char_p(output_layout.value).value
    ######    model_name = c_char_p(mode_name.value).value
    ######       
    ######    encoding = 'utf-8'
    ######    args_dict = {
    ######        'xclbin'     : xclbin,
    ######        'netcfg'     : str(path + model_name + b"_compiler.json"  ,encoding),
    ######        'quantizecfg': str(path + model_name + b"_quantizer.json" ,encoding),
    ######        'weights'    : str(path + model_name + b"_weights.h5"     ,encoding), #config['weights'] + '_data.h5' if config['weights'] else "",
    ######        'scaleA'     : 1,
    ######        'scaleB'     : 1,
    ######        'PE'         : 0,
    ######        'batch_sz'   : ins[0].shape[0],
    ######        'inshape'    : tuple(ins[0].shape[1:])
    ######    }
     
    ret, handles = xdnn.createHandle(xclbin)
     
    if ret != 0:
        print("ERROR: Unable to create handle to FPGA")
        return
    else:
        print("INFO: Successfully created handle to FPGA")
     
 
     
    print("PATH=",path)
    args = xdnn_io.make_dict_args(args_dict)
     
    fpgaRT = xdnn.XDNNFPGAOp(handles,args)
     
    fpgaInput = fpgaRT.getInputs()
    fpgaOutput = fpgaRT.getOutputs()
     
    batch_array = np.empty(((ins[0].shape[0],) + tuple(ins[0].shape[1:])), dtype=np.float32, order='C')
    data_paths = [ins[0].asnumpy()]
     
    for i in range(0, len(data_paths), ins[0].shape[0]):
        for j, d in enumerate(data_paths[i:i + ins[0].shape[0]]):
            batch_array[j, ...] = d
            
    ##    print(batch_array)
     
    # TODO HAS TO BE CHANGED FOR MULTIPLE INPUTS
    fpgaInput[list(fpgaInput.keys())[0]] = batch_array
    ##    print(fpgaInput)
     
    # WRITE FPGA INSTRUCTIONS TO FPGA AND EXECUTE THE NETWORK!
    #print(fpgaOutput)
    fpgaRT.execute(fpgaInput, fpgaOutput)
     
     
    # GET OUTPUT
    key, value  = fpgaOutput.popitem()
     
    # DEFAULT FPGA OUTPUT LAYOUT IS NCHW
    if str(layout) == 'NHWC':
        value = np.transpose(value,(0,2,3,1))


        
    tvm.nd.array(value).copyto(out)
     
    print(" -- debug: tvm_reg_func done ")



