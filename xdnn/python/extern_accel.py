"""xDNN implementation of NN operations that can be accelerated"""


import tvm
import numpy as np
import pdb
from ctypes import *
import ctypes
import os
import warnings

# TEMP
try:
    import xfdnn.rt.xdnn as xdnn
    import xfdnn.rt.xdnn_io as xdnn_io
    from xfdnn.rt import xdnn, xdnn_io
except:
    warnings.warn("Could not import xfdnn libraries")

try:
    from dnndk import n2cube, dputils
except:
    warnings.warn("Could not import dnndk n2cube")
    
# TODO: ADD MODEL NAME FOR NNVM
@tvm.register_func("tvm.accel.accel_fused")
def accel_fused(graph_path, output_layout, model_name, platform, out, *ins ):

    #pdb.set_trace()
    # TEMP
    print(platform)


    if platform == "DPU":


        kernel_name = "xp0"
        input_name  = "xinput0"
        output_name = "resnet_v1_50/logits/Conv2D"

        """ Attach to DPU driver and prepare for running """
        n2cube.dpuOpen()
        
        """ Create DPU Kernels for CONV NODE in imniResNet """
        kernel = n2cube.dpuLoadKernel(kernel_name)

        """ Create DPU Tasks for CONV NODE in miniResNet """
        task = n2cube.dpuCreateTask(kernel, 0)


        """ Load image to DPU in (CHW or HWC format) """
        n2cube.dpuSetInputTensorInHWCFP32(task, input_name, ins[0], len(ins[0]))

        """ Model run on DPU """
        n2cube.dpuRunTask(task)
        
        """ Get the output tensor size from FC output """
        size = n2cube.dpuGetOutputTensorSize(task, output_name)
        address = n2cube.dpuGetOutputTensorAddress(task, output_name)
        

        value = [0 for i in range(size)]

        n2cube.dpuGetTensorData (address, value, size)
        scale = n2cube.dpuGetOutputTensorScale(task, output_name, idx=0)
        value = np.array(value).astype(np.float32)/scale
        

        # DEFAULT DPU OUTPUT IS NHWC
        if layout == 'NCHW':
            value = np.transpose(value,(0,3,1,2))

        
    else:
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



