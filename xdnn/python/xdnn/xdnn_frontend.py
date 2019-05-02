"""xDNN frontend class"""

from xdnn import XDNNError
from xdnn.xdnn_controller import XDNNController

from xfdnn.tools.compile.bin.xfdnn_compiler_tvm import TVMFrontend

class XDNNFrontend(object):

    """
    Parameters
    ----------
    ...
    """
    def __init__(self):
        #
        self.platform = None
        self.xclbin = None
        self.memory = None
        self.dsp = None
        self.netcfg = None
        self.weightsdir_name = None
        self.quantizecfg = None
        self.pngfile = None
        self.verbose = False

        self.tvm_compiler = None
        self.xdnn_controller = None

    def check_initialized(self):
        # () -> bool
        if not isinstance(self.tvm_compiler, TVMFrontend)\
            or not isinstance(self.xdnn_controller, XDNNController):
            raise XDNNError("Initialize xDNN frontend before calling main methods"
                " (compile, quantize, execute)")
        return True

    def init(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'platform':
                    self.set_platform(value)
                elif key == 'xclbin':
                    self.set_xclbin(value)
                elif key == 'memory':
                    self.set_memory(value)
                elif key == 'dsp':
                    self.set_dsp(value)
                elif key == 'netcfg':
                    self.set_netcfg(value)
                elif key == 'weightsdir_name':
                    self.set_weightsdir_name(value)
                elif key == 'quantizecfg':
                    self.set_quantizecfg(value)
                elif key == 'pngfile':
                    self.set_pngfile(value)
                elif key == 'verbose':
                    self.verbose = value

        if self.platform != None and self.xclbin != None and self.memory != None\
            and self.dsp != None and self.netcfg != None and self.weightsdir_name != None:
            
            self.tvm_compiler = TVMFrontend(
                networkfile="None",
                memory=self.memory,
                dsp=self.dsp,
                # Because tvm compiler will write FPGA code json to this file,
                # but we will construct or own file: netcfg attribute
                generatefile= "temp_fpga_code.cmds", 
                fromtensorflow=False,
                weights=self.weightsdir_name,
                pngfile=self.pngfile,
                verbose=self.verbose
            )

            self.xdnn_controller = XDNNController(
                platform=self.platform,
                xclbin=self.xclbin,
                memory=self.memory,
                dsp=self.dsp,
                netcfg=self.netcfg,
                datadir="work/" + self.weightsdir_name + "_data",
                quantizecfg=self.quantizecfg
            )
        else:
            raise XDNNError("One of following mandatory arguments is missing:"\
                "platform, xclbin, memory, dsp, netcfg, datadir")

    ## MAIN METHODS ##
    
    def compile(self, op, name, attrs, inputs, shapes, layout, params):
        # type: (str, str, Dict[str, str], List[str], List[List[int]], str,
        #    Dict[str,tvm.ndarray.NDArray]) -> json

        fpga_code_json = self.tvm_compiler.compile_op(op, name, attrs, 
            inputs, shapes, layout, params)

        print(type(fpga_code_json))
        print("Compiled json code: {}".format(fpga_code_json))
        
        self.xdnn_controller.add_operation(name, fpga_code_json)
        
        return "SUCCESS"

    def quantize(self):
        # TODO: how/where to do quantization
        raise NotImplementedError("")

    def execute(self, name, ins, outs):
        # (str) -> None
        """
        Execute the operation with the given name and inputs and write result to 
        output data structure
        """
        # TODO: simulator
        self.xdnn_controller.execute_op(name, ins, outs)

    def setup_fpga_executer(self):
        # type: () -> None
        """
        Setting up the connecion with the FPGA and retrieving an xDNN executer instance
        """
        self.xdnn_controller.set_fpga_rt()

    ## GETTERS & SETTERS ##
    # TODO: checks

    def set_platform(self, platform):
        # type: (str) -> None
        self.platform = platform

    def set_xclbin(self, xclbin):
        # (str) -> None
        self.xclbin = xclbin

    def set_memory(self, memory):
        # type: (int) -> None
        self.memory = memory

    def set_dsp(self, dsp):
        # type: (int) -> None
        self.dsp = dsp

    def set_netcfg(self, netcfg):
        # type: (str) -> None
        if not netcfg.endswith('.json'):
            raise XDNNError("Invalid netcfg file, file name should end with .json")
        self.netcfg = netcfg

    def set_weightsdir_name(self, weightsdir_name):
        # type: (str) -> None
        self.weightsdir_name = weightsdir_name

    def set_quantizecfg(self, quantizecfg):
        # type: (str) -> None
        self.quantizecfg = quantizecfg

    def set_pngfile(self, pngfile):
        # type: (str) -> None
        self.pngfile = pngfile


    

