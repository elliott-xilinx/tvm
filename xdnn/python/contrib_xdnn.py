"""xDNN implementation of NN operations that can be accelerated"""


import tvm
import numpy as np
import pdb
from nnvm.top import registry as reg
from ctypes import *
import ctypes
import os

import xfdnn.rt.xdnn as xdnn
import xfdnn.rt.xdnn_io as xdnn_io
#from xfdnn.rt import xdnn, xdnn_io

@reg.register_schedule("xdnn", level=15)
def schedule_xdnn(attrs,outputs,target):
    #pdb.set_trace()
    print("-- debug: xdnn schedule")
    return tvm.create_schedule([x.op for x in outputs])


    
@reg.register_compute("xdnn", level=15)
def compute_xdnn(attrs,inputs,outputs):
    print ("-- debug: xdnn compute")

    op = 'xdnn'
    name = 'xdnn0'
    attrs_dict = { k: attrs[k] for k in attrs.keys() }
    input_names = [inpt.op.name for inpt in inputs]
    in_shapes = [[int(i) for i in inpt.shape] for inpt in inputs]
    out_shapes = [[int(i) for i in outputs[0].shape]]

    # EXTERNAL FUNCTION TO RUN THE FUSED OPERATION
    #out = tvm.extern(outputs[0].shape, inputs, lambda ins, outs: tvm.call_packed('tvm.xdnn.xdnn', ins[0], ins[1], ins[1], outs[0], name), name=name)

    #pdb.set_trace()

    out = tvm.extern(outputs[0].shape, inputs, lambda ins, outs: tvm.call_packed('tvm.xdnn.xdnn_fused', attrs['json_graph'], outs[0], *ins ), name=name)
    
      
    print(out.shape)
    
    return out
    #return outputs
    
    

@tvm.register_func("tvm.xdnn.xdnn_fused")
#def xdnn_fused(data, weights, bias, out, name ):
#def xdnn_fused(data, out, name ):
def xdnn_fused(graph_path, out, *ins ):

    #pdb.set_trace()

    path = c_char_p(graph_path.value).value
    # CREATE A HANDLE FOR FPGA COMMUNICATION
    platform = os.environ.get('MLSUITE_PLATFORM')#"alveo-u200"
    xclbin = "/workspace/MLsuite/overlaybins/" + platform + "/overlay_4.xclbin"
    #xdnn_mgr = xdnn.XDNNManager()
    print("created xdnn manager")
    #ret, handles = xdnn_mgr.createHandle(xclbin, "kernelSxdnn_0")
    ret, handles = xdnn.createHandle(xclbin)
    
    print("Handle value: %s" % handles[0].value)
    #assert ret==0, print("ERROR: Unable to create handle to FPGA")
    if ret != 0:
        print("ERROR: Unable to create handle to FPGA")
    else:
        print("INFO: Successfully created handle to FPGA")

    encoding = 'utf-8'
    args_dict = {
        'xclbin': xclbin,
        'netcfg': str(path + b"/work/tvm_compiler.json",encoding),
        'quantizecfg': str(path +  b"/work/tvm_quantizer.json",encoding),
        'weights': str(path + b'/work/' + b'weights_data.h5',encoding), #config['weights'] + '_data.h5' if config['weights'] else "",
        'scaleA': 1,
        'scaleB': 1,
        'PE': 0,
        'batch_sz': ins[0].shape[0],
        'inshape': tuple(ins[0].shape[1:])
    }
    
    args = xdnn_io.make_dict_args(args_dict)
    print(args)

    fpgaRT = xdnn.XDNNFPGAOp(handles,args)

    fpgaInput = fpgaRT.getInputs()
    fpgaOutput = fpgaRT.getOutputs()
    print(fpgaInput)
    print(fpgaOutput)

    batch_array = np.empty(((ins[0].shape[0],) + tuple(ins[0].shape[1:])), dtype=np.float32, order='C')
    data_paths = [ins[0].asnumpy()]

    for i in range(0, len(data_paths), ins[0].shape[0]):
        for j, d in enumerate(data_paths[i:i + ins[0].shape[0]]):
            batch_array[j, ...] = d
            
    print(batch_array)

    # TEMP IS THIS NECESSARY?
    #fpgaInput[input_names[0]] = batch_array
    print(fpgaInput)

    # Write FPGA Instructions to FPGA and Execute the network!
    print(fpgaOutput)
    fpgaRT.execute(fpgaInput, fpgaOutput)
    print(fpgaOutput)

    
    #temp = tvm.nd.array(np.zeros(out.shape, dtype=np.float32, order='C'))
    #tvm.nd.array(fpgaOutput['max_pool2d0']).copyto(out)
    key, value  = fpgaOutput.popitem()
    tvm.nd.array(value).copyto(out)
    print(" -- debug: tvm_reg_func done ")
    #temp.copyto(out)
    
