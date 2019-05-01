"""
xDNN acceleration backend

Implementation of FPGA accelerated operations
"""

class XDNNError(Exception):
    """Error throwed by xDNN functions"""
    pass

from xdnn.xdnn_frontend import XDNNFrontend

xdnn_frontend = XDNNFrontend()

import xdnn.nn_def
import xdnn.nn_impl








