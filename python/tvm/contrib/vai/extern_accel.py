# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

""" Implementation of external Vitis-AI accel operation """

import os
import tvm
import warnings
import numpy as np

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


@tvm.register_func("tvm.accel.accel_fused")
def accel_fused(kernel_name, input_name, output_name, layout, out, *ins):

    # Attach to DPU driver and prepare for running
    n2cube.dpuOpen()
    
    # Create DPU Kernels
    kernel = n2cube.dpuLoadKernel(kernel_name)

    # Create DPU Tasks for kernel
    task = n2cube.dpuCreateTask(kernel, 0)

    # Load image to DPU
    X = ins[0].asnumpy().reshape((-1))
    n2cube.dpuSetInputTensorInHWCFP32(task, input_name, X, len(X))

    # Model run on DPU """
    n2cube.dpuRunTask(task)
    
    # Get the output tensor size 
    size = n2cube.dpuGetOutputTensorSize(task, output_name)
    address = n2cube.dpuGetOutputTensorAddress(task, output_name)

    value = [0 for i in range(size)]

    n2cube.dpuGetTensorData (address, value, size)
    scale = n2cube.dpuGetOutputTensorScale(task, output_name, idx=0)
    value = np.array(value).astype(np.float32)/scale

    # DPU output is in NHWC but graph is executed in NCHW
    if output_layout == 'NCHW':
        value = np.transpose(value,(0,3,1,2))

    tvm.nd.array(value).copyto(out)

