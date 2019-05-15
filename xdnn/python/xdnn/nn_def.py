"""xDNN definition of NN operations that can be accelerated"""

import tvm
import nnvm

from xdnn import XDNNError, xdnn_frontend

## CONVOLUTION ##

def compute_conv2d(attrs, inputs, outputs):
    """Compute definition of conv2d
    
    Parameters
    ----------
    attrs : nnvm.top.attr_dict.AttrDict
        dictionary of operation attributes 

    inputs : List[tvm.Tensor]
        list of operation input tensors

    outputs : List[tvm.Tensor]
        list of operation output tensor placeholders

    Returns
    -------
    out : tvm.Tensor
        the output tensor
    """
    
    op = 'conv2d'
    name = 'conv2d0'
    attrs_dict = { k: attrs[k] for k in attrs.keys() }
    input_names = [inpt.op.name for inpt in inputs]
    print(input_names)
    in_shapes = [[int(i) for i in inpt.shape] for inpt in inputs]
    out_shapes = [[int(i) for i in outputs[0].shape]]
    shapes = in_shapes + out_shapes
    layout = attrs_dict['layout']
    params = {}

    xdnn_frontend.check_initialized()
    xdnn_frontend.compile(op, name, attrs_dict, input_names, shapes, layout, params)
    
    # TODO: check that we can execute this conv layer on fpga, otherwise return topi definition

    I, O = inputs[0], outputs[0] 
    # Construct TVM external function call for computing 2d convolution on FPGA
    out = tvm.extern(O.shape, [I], lambda ins, outs: tvm.call_packed(
        'tvm.xdnn.conv2d', ins[0], outs[0], name
        ), name=name)
    
    # TODO: checks, layout
    # out = topi.nn.pool(inputs[0], kernel, stride, padding, 
    #                    pool_type='max', layout=layout)

    print("conv2d out: {}".format(out))
    print(out.shape)
    return out

# level should be higher than 10 to override nnvm max_pool2d computation definition
nnvm.top.register_compute('conv2d', compute_conv2d, level=15) 

def schedule_conv2d(attrs, outputs, target):
    """Schedule definition of conv2d
    
    Parameters
    ----------
    attrs : dict
        dictionary of operation attributes 

    outputs : List[tvm.Tensor]
        list of operation output tensor placeholders

    target: str
        the target device identifier

    Returns
    -------
    s : tvm.Schedule
        the operation schedule
    """
    return tvm.create_schedule([x.op for x in outputs])

nnvm.top.register_schedule('conv2d', schedule_conv2d, level=15)

## POOLING ##

def compute_max_pool2d(attrs, inputs, outputs):
    """Compute definition of max_pool2d
    
    Parameters
    ----------
    attrs : nnvm.top.attr_dict.AttrDict
        dictionary of operation attributes 

    inputs : List[tvm.Tensor]
        list of operation input tensors

    outputs : List[tvm.Tensor]
        list of operation output tensor placeholders

    Returns
    -------
    out : tvm.Tensor
        the output tensor
    """

    print(type(attrs))
    
    op = 'max_pool2d'
    name = 'max_pool2d0'
    attrs_dict = { k: attrs[k] for k in attrs.keys() }
    input_names = [inpt.op.name for inpt in inputs]
    print(input_names)
    in_shapes = [[int(i) for i in inpt.shape] for inpt in inputs]
    out_shapes = [[int(i) for i in outputs[0].shape]]
    shapes = in_shapes + out_shapes
    layout = attrs_dict['layout']
    params = {}

    xdnn_frontend.check_initialized()
    xdnn_frontend.compile(op, name, attrs_dict, input_names, shapes, layout, params)
    
    """
    kernel_h, kernel_w = attrs.get_int_tuple("pool_size")
    stride_h, stride_w, = attrs.get_int_tuple("strides")
    padding = attrs.get_int_tuple("padding")
    layout = attrs['layout']
    ceil_mode = attrs['ceil_mode']
    
    if len(padding) == 1:
        padding = padding * 4
    elif len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    elif len(padding) != 4:
        raise XDNNError("Invalid number of paddings for 2d max pool operation,"\
            " expected 4 but got: {}".format(len(padding)))
    
    if len(inputs) != 1:
        raise XDNNError("Invalid number of inputs for 2d max pool operation,"\
            " expected 1 but got: {}".format(len(inputs)))

    pad_t, pad_l, pad_b, pad_r = padding
    """
    # TODO: check that we can execute this maxpool layer on fpga, otherwise return topi definition

    I, O = inputs[0], outputs[0] 
    # Construct TVM external function call for computing 2d max pool on FPGA
    out = tvm.extern(O.shape, [I], lambda ins, outs: tvm.call_packed(
        'tvm.xdnn.max_pool2d', ins[0], outs[0], name
        # kernel_h, kernel_w, stride_h, stride_w, pad_t, pad_l, pad_b, pad_r, ceil_mode, layout
        ), name=name)
    
    # TODO: checks, layout
    # out = topi.nn.pool(inputs[0], kernel, stride, padding, 
    #                    pool_type='max', layout=layout)

    print("Max pool out: {}".format(out))
    print(out.shape)
    return out

# level should be higher than 10 to override nnvm max_pool2d computation definition
nnvm.top.register_compute('max_pool2d', compute_max_pool2d, level=15)   

def schedule_max_pool2d(attrs, outputs, target):
    """Schedule definition of max_pool2d
    
    Parameters
    ----------
    attrs : dict
        dictionary of operation attributes 

    outputs : List[tvm.Tensor]
        list of operation output tensor placeholders

    target: str
        the target device identifier

    Returns
    -------
    s : tvm.Schedule
        the operation schedule
    """
    # layout = attrs['layout']
    # with tvm.target.create(target):
    #    return topi.generic.schedule_pool(outs, layout=layout)
    
    # just use the basic tvm create_schedule function
    # print([x.op for x in outputs])
    return tvm.create_schedule([x.op for x in outputs])

nnvm.top.register_schedule('max_pool2d', schedule_max_pool2d, level=15)