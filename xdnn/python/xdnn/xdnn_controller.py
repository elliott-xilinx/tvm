"""xDNN IP controller class"""

import time
import json
import os.path

from xdnn import XDNNError

import xfdnn.rt.xdnn as xdnn_lib
import xfdnn.rt.xdnn_io as xdnn_io

# TODO: move
class XDNNOp(object):

    """
    """

    def __init__(self, netcfg_json):
        # type: (dict) -> XDNNOp
        inputs = netcfg_json['inputs']
        self.input_names = [inpt['input_name'] for inpt in inputs]


class XDNNController(object):

    """
    This class wraps the FPGA backend and controls requests

    Attributes
    ----------
    handles: list of pointers

    op_to_lines: dict
        dictionary mapping operation names to start and end line numbers in
        command file
    """

    # TODO: Singleton class??
    """
    __instance = None

    @classmethod
    def get_instance(self):
        if XDNNController.__instance is not None:
            return XDNNController.__instance
        return XDNNController()
    """

    def __init__(self, platform, xclbin, netcfg, params_loc, input_shape, 
            quantizecfg=None, scaleA=1, scaleB=1, PE=0):
        """
        if XDNNController.__instance is not None:
            raise XDNNError("XDNNController is a singleton class and only one"\
                "instance can be created. Call get_instance to get this instance.")
        else:
            XDNNController.__instance = self
        """
        self.platform = platform
        self.xclbin = xclbin
        #self.memory = memory
        #self.dsp = dsp
        self.netcfg = netcfg
        if os.path.isfile(self.netcfg):
            os.remove(self.netcfg)
        self.params_loc = params_loc
        self.quantizecfg = quantizecfg
        # TODO: allow custom values
        self.scaleA = scaleA
        self.scaleB = scaleB
        self.PE = PE
        self.input_shape = input_shape
        #self.batch_sz = 1
        #self.in_shape = (1,4,4)
        # TODO: batch_sz, in_shape??

        self.ops = {}
        self.op_to_lines = {}

        self.handles = None
        self.fpga_rt = None
        self.fpga_input = {}
        self.fpga_output = {}
    
    def execute_op(self, name, ins):
        # (str) -> None
        """
        Execute the operation with the given name and inputs and write result to 
        output data structure
        """
        if self.fpga_rt is None:
            raise ValueError("Setup FPGA executer before executing the graph")
        print(ins.shape, type(ins))
        print(ins)
        #print(outs.shape, type(outs))
        #print(outs)
        # TODO:
        xdnn_op = self.ops[name]
        input_name = xdnn_op.input_names[0]
        self.fpga_input[input_name] = ins
        
        self.fpga_rt.execute(self.fpga_input, self.fpga_output)
        print("after")
        print(self.fpga_output)

        return self.fpga_output[name]


    def add_operation(self, op_name, netcfg_json, quant_params=[]):
        # (str, str, List[dict]) -> None
        """
        Add an operation code to command file, map operation name to start/end
        line numbers in command file and add quant params to quantization 
        parameters file 
        """
        print("Add operation: {}".format(op_name))
        xdnn_op = XDNNOp(netcfg_json)
        self._add_op(op_name, xdnn_op)

        base_netcfg_json = self.get_netcfg_json()
        if base_netcfg_json is None:
            self.set_netcfg_json(netcfg_json)
        else:
            # TODO: update instruction indices???
            # TODO: Update ops
            for op in netcfg_json['network']:
                base_netcfg_json['network'].append(op)

            # TODO: Update supported count
            for name, instr_lst in netcfg_json['unsupported']['list'].items():
                base_netcfg_json['unsupported']['list'][name] = instr_lst

            self.set_netcfg_json(base_netcfg_json)

    ## GETTERS & SETTERS ##

    def set_fpga_rt(self):
        # type: () -> None
        """
        Setting up the connecion with the FPGA and initializing with the set configs
        to retrieve a xDNN executer instance
        """
        if self.fpga_rt is not None:
            return self.fpga_rt
        elif not self.xclbin:
            raise XDNNError("Specify xclbin file by initizialing before creating"\
                " an FPGA handle")
        if not os.path.isfile(self.params_loc):
            print("WARNING parameters location does not exist when setting up fpga"\
                  " configuration")
            self.params_loc = ""
        
        args_dict = {
                'xclbin': self.xclbin,
                'netcfg': self.netcfg,
                'quantizecfg': self.quantizecfg,
                'weights': self.params_loc,
                'scaleA': self.scaleA,
                'scaleB': self.scaleB,
                'PE': self.PE,
                'batch_sz': self.input_shape[0],
                'inshape': tuple(self.input_shape[1:])
            }
        args = xdnn_io.make_dict_args(args_dict)
        print("Set_fpga_rt: args: {}".format(args))

        print("Create handle")        
        ret, handles = xdnn_lib.createHandle(self.xclbin)
        print("after create handle")
        #ret = False
        if ret:                                                         
            print("ERROR: Unable to create handle to FPGA")

            # TODO: simulator
        else:
            print("INFO: Successfully created handle to FPGA")
            
            self.fpga_rt = xdnn_lib.XDNNFPGAOp(handles, args)
            self.fpga_input = fpga_rt.getInputs()
            self.fpga_output = fpga_rt.getOutputs()


    ## HELPER METHODS ##

    def _add_op(self, op_name, xdnn_op):
        # type: (str, XDNNOp) -> None
        # TODO: make a class
        if op_name in self.ops:
            raise XDNNError("Operation with name: {} already exists, "\
                "duplicate error".format(op_name))
        self.ops[op_name] = xdnn_op

    def get_netcfg_json(self):
        # Load existing json from netcfg file
        if os.path.isfile(self.netcfg):
            with open(self.netcfg) as f:
                return json.load(f)
        return None

    def add_to_netcfg_json(self, new_netcfg_json):
        # type: (dict) -> None
        pass

    def set_netcfg_json(self, netcfg_json):
        # type: (dict) -> None
        with open(self.netcfg, 'w') as f:
            f.write(json.dumps(netcfg_json, indent=4, sort_keys=True))
