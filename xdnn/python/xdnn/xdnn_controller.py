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

    def __init__(self, name, netcfg_json):
        # type: (str, dict) -> XDNNOp
        self.name = name
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

        self.op_idx = 0
        self.ops = {}
        self.op_to_lines = {}

        self.handles = None
        self.fpga_rt = None
        self.fpga_input = {}
        self.fpga_output = {}
    
    def execute_op(self, op_id, ins):
        # type: (int, numpy.ndarray) -> numpy.ndarray
        """
        Execute the operation with the given name and inputs and return the result
        """
        if self.fpga_rt is None:
            raise ValueError("Setup FPGA executer before executing the graph")

        xdnn_op = self.ops[op_id]
        input_name = xdnn_op.input_names[0]
        print("Execute op id: {}, name: {}, ins shape: {}".format(op_id, xdnn_op.name, ins.shape))

        self.fpga_input[input_name] = ins
        
        print("FPGA input: {}".format(self.fpga_input))
        print("FPGA output before: {}".format(self.fpga_output))
        self.fpga_rt.execute(self.fpga_input, self.fpga_output)
        print("FPGA output after: {}".format(self.fpga_output))

        return self.fpga_output[xdnn_op.name]


    def add_operation(self, op_id, op_name, netcfg_json, quant_params=[]):
        # type: (int, str, str, List[dict]) -> None
        """
        Add an operation code to command file, map operation name to start/end
        line numbers in command file and add quant params to quantization 
        parameters file 
        """
        print("Add operation: {}".format(op_id))
        xdnn_op = XDNNOp(op_name, netcfg_json)

        self._add_op(op_id, xdnn_op)

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
               
        ret, handles = xdnn_lib.createHandle(self.xclbin)

        #ret = False
        if ret:                                                         
            print("ERROR: Unable to create handle to FPGA")

            # TODO: simulator
        else:
            print("INFO: Successfully created handle to FPGA")
            
            self.fpga_rt = xdnn_lib.XDNNFPGAOp(handles, args)
            self.fpga_input = self.fpga_rt.getInputs()
            self.fpga_output = self.fpga_rt.getOutputs()


    ## HELPER METHODS ##

    def _add_op(self, op_id, xdnn_op):
        # type: (str, XDNNOp) -> None
        # TODO: make a class
        if op_id in self.ops:
            raise XDNNError("Operation with id: {} already exists, "\
                "duplicate error".format(op_id))
        self.ops[op_id] = xdnn_op
        self.op_idx += 1

    def get_new_op_id(self):
        # type: () -> int
        return self.op_idx

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
