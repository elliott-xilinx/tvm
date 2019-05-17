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
    
    op_id = xdnn_frontend.get_new_op_id()
    op = 'conv2d'
    name = str(op_id) + "_" + op
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

    I, O = inputs[0], outputs[0] 
    # Construct TVM external function call for computing 2d convolution on FPGA
    out = tvm.extern(O.shape, [I], lambda ins, outs: tvm.call_packed(
        'tvm.xdnn.conv2d', ins[0], outs[0], name
        ), name=name)
    
    # TODO: checks, layout
    # out = topi.nn.pool(inputs[0], kernel, stride, padding, 
    #                    pool_type='max', layout=layout)

    print("Conv2d: {} , out shape: {}".format(name, out.shape))
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
    
    op_id = xdnn_frontend.get_new_op_id()
    op = 'max_pool2d'
    name = str(op_id) + "_" + op 
    
    attrs_dict = { k: attrs[k] for k in attrs.keys() }
    input_names = [inpt.op.name for inpt in inputs]
    print(input_names)
    in_shapes = [[int(i) for i in inpt.shape] for inpt in inputs]
    out_shapes = [[int(i) for i in outputs[0].shape]]
    shapes = in_shapes + out_shapes
    layout = attrs_dict['layout']
    params = {}

    xdnn_frontend.check_initialized()
    xdnn_frontend.compile(op, op_id, name, attrs_dict, input_names, shapes, layout, params)

    I, O = inputs[0], outputs[0] 
    # Construct TVM external function call for computing 2d max pool on FPGA
    print("name: {}".format(name))
    out = tvm.extern(O.shape, [I], lambda ins, outs: tvm.call_packed(
        'tvm.xdnn.max_pool2d', ins[0], outs[0], op_id
        ), name=name)
    
    # TODO: checks, layout
    # out = topi.nn.pool(inputs[0], kernel, stride, padding, 
    #                    pool_type='max', layout=layout)

    print("Max pool: {} , out shape: {}".format(name, out))
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
    # just use the basic tvm create_schedule function
    # print([x.op for x in outputs])
    return tvm.create_schedule([x.op for x in outputs])

nnvm.top.register_schedule('max_pool2d', schedule_max_pool2d, level=15)
