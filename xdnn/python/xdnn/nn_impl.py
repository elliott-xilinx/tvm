"""xDNN implementation of NN operations that can be accelerated"""


import tvm
import numpy as np
import ctypes

from xdnn import XDNNError, xdnn_frontend

@tvm.register_func("tvm.xdnn.conv2d")
def xdnn_conv2d(inpt, out, name):
    # (tvm.ndarray.NDArray, tvm.ndarray.NDArray, str) -> None
    """
    Implementation of FPGA accelerated conv2d operation

    Returns:
        None, operation of operation will be written to 'out' data structure
    """
    print(inpt, type(inpt), inpt.shape)
    print(out, type(out), out.shape)
    #res = np.reshape(np.array([[5,7],[9,4]], dtype=np.float32), (1,1,2,2))
    
    fpga_output = np.empty(out.shape, dtype=np.float32, order='C')
    print(fpga_output)

    xdnn_frontend.execute(name, inpt.asnumpy(), fpga_output)

    print("fpga out")
    print(fpga_output)

    tvm.nd.array(fpga_output).copyto(out)
    print(out)
    print("after outs")


@tvm.register_func("tvm.xdnn.max_pool2d")
def xdnn_max_pool2d(inpt, 
                    out, 
                    op_id):
    # (tvm.ndarray.NDArray, tvm.ndarray.NDArray, int, int, int, int,
    #   int, int, int, int, str) -> None
    """
    Implementation of FPGA accelerated max pool operation

    Returns:
        None, operation of operation will be written to 'out' data structure
    """
    print("maxpool2d: op_id: {}, type: {}".format(op_id, type(op_id)))
    #print(ctypes.cast(s.value, ctypes.py_object).value)
    #res = np.reshape(np.array([[5,7],[9,4]], dtype=np.float32), (1,1,2,2))
    
    #fpga_output = np.empty(out.shape, dtype=np.float32, order='C')
    #print(fpga_output)

    fpga_output = xdnn_frontend.execute(op_id, inpt.asnumpy())

    print("fpga out")
    print(fpga_output)

    tvm.nd.array(fpga_output).copyto(out)
    print(out)
    print("after outs")
