import numpy as np
import tvm
import extern_accel
from cvx.img_loader import ImgLoader
from cvx.img_processor import ImgProcessor

data_inputs = ['Placeholder']
input_formats = ['NHWC']
data_shapes = {'Placeholder': [1,224,224,3]}
#data_io = "resize_smallest_side-256__central_crop-224-224-3__subtract-123.68,116.78,103.94__transpose-2,0,1"
data_io = "resize_smallest_side-256__central_crop-224-224-3__subtract-123.68,116.78,103.94"

if input_formats[0] =='NCHW':
    input_shape = data_shapes[data_inputs[0]]
elif input_formats[0] =='NHWC':
    ishape = data_shapes[data_inputs[0]]
    input_shape = data_shapes[data_inputs[0]]
    #input_shape = [ishape[0], ishape[3], ishape[1], ishape[2]]
else:
    raise NotImplementedError("")


# LOADING & PREPROCESSING #
img_loader = ImgLoader(
    layout = 'NHWC',
    color_format = 'RGB'
)
data_preprocessor = ImgProcessor(
    proc_key = data_io
)

img_paths = ["/home/xilinx/xfgraph/examples/images/dog.jpg"] # Image of interest (Must provide as a list)
val_set = [["/home/xilinx/xfgraph/examples/images/dog.jpg", 259]]

batch_array = np.empty([len(img_paths)]+input_shape[1:], dtype=np.float32, order='C')
data = img_loader.load(img_paths)
preprocessed_data = data_preprocessor.execute(data)
batch_array[:] = preprocessed_data



# RUN #

inputs = {}
inputs[list(data_shapes.keys())[0]] = batch_array

import pdb
pdb.set_trace()
tvm_lib    = tvm.module.load("/home/xilinx/tvm-dpu/examples/tvm_dpu_cpu.so")
tvm_json   = open("/home/xilinx/tvm-dpu/examples/tvm_dpu_cpu.json").read()
tvm_params = bytearray(open("/home/xilinx/tvm-dpu/examples/tvm_dpu_cpu.params", "rb").read())
 
# graph runtime initializes runtime from loaded module,
from tvm.contrib import graph_runtime
module = graph_runtime.create(tvm_json, tvm_lib, tvm.cpu(0))
 
# INITIALIZE THE PARAMETERS.

module.load_params(tvm_params)
module.set_input(**inputs)

# RUN

module.run()
res = module.get_output(0)
res = res.asnumpy()

res.shape
repr(res)
np.max(res)
np.min(res)
np.argmax(res)

# PREDICTIONS #
from xfgraph import classification

raw_predictions = res
label_lst = [elem[1] for elem in val_set[:raw_predictions.shape[0]]]
synset_words = '/home/xilinx/xfgraph/examples/images/imagenet_synset_words.txt'

top_1 = classification.get_top_k_accuracy(raw_predictions, synset_words, 1, label_lst)
top_5 = classification.get_top_k_accuracy(raw_predictions, synset_words, 5, label_lst)   
print("Top 1: {}".format(top_1))
print("Top 5: {}".format(top_5))
