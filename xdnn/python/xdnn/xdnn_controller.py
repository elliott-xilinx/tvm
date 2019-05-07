"""xDNN IP controller class"""

import time
import json
import os.path

from xdnn import XDNNError

import xfdnn.rt.xdnn as xdnn_lib

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

    def __init__(self, platform, xclbin, memory, dsp, netcfg, datadir, quantizecfg=None):
        """
        if XDNNController.__instance is not None:
            raise XDNNError("XDNNController is a singleton class and only one"\
                "instance can be created. Call get_instance to get this instance.")
        else:
            XDNNController.__instance = self
        """
        self.platform = platform
        self.xclbin = xclbin
        self.memory = memory
        self.dsp = dsp
        self.netcfg = netcfg
        if os.path.isfile(self.netcfg):
            os.remove(self.netcfg)
        self.datadir = datadir
        self.quantizecfg = quantizecfg

        # TODO: allow custom values
        self.scaleA = 1
        self.scaleB = 1
        self.PE = 0
        self.batch_sz = 1
        self.in_shape = (1,4,4)
        # TODO: batch_sz, in_shape??

        self.handles = None
        self.fpga_rt = None
        self.op_to_lines = {}
    
    def execute_op(self, name, ins, outs):
        # (str) -> None
        """
        Execute the operation with the given name and inputs and write result to 
        output data structure
        """
        if self.fpga_rt is None:
            raise ValueError("Setup FPGA executer before executing the graph")
        print(ins.shape, type(ins))
        print(ins)
        print(outs.shape, type(outs))
        print(outs)
        
        self.fpga_rt.execute(ins, outs)
        print("after")
        print(outs)

    def add_operation(self, op_name, netcfg_json, quant_params=[]):
        # (str, str, List[dict]) -> None
        """
        Add an operation code to command file, map operation name to start/end
        line numbers in command file and add quant params to quantization 
        parameters file 
        """
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
        print("Create handle")        
        ret, handles = xdnn_lib.createHandle(self.xclbin)
        print("after create handle")
        #ret = False
        if ret:                                                         
            print("ERROR: Unable to create handle to FPGA")
        else:
            print("INFO: Successfully created handle to FPGA")
            print("test")
            
            config = {
                #'xclbin': self.xclbin,
                #'platform':self.platform,
                #'networkfile': 'None',
                'memory': self.memory,
                'dsp': self.dsp,
                'netcfg': self.netcfg[:-5], # TODO: remove .json ??
                #'fromtensorflow': False,
                'weights': "weights", 
                'datadir': self.datadir,
                'pngfile': "graph.png",
                'verbose': True,
                'quantizecfg': "", #"work/tvm_quantization_params.json", #TODO
                'img_mean': [100],
                'calibration_size': 15,
                'bitwidths': [16,16,16],
                'img_raw_scale': 255.0,
                'img_input_scale': 1.0,
                # FPGA
                'scaleA': self.scaleA,
                'scaleB': self.scaleB,
                'PE': self.PE,
	        'batch_sz': self.batch_sz,
	        'in_shape': self.in_shape
                # TODO: batch_sz, in_shape??
            }
            #config2 = {'networkfile': 'None', 'memory': 5, 'dsp': 56, 'netcfg': 'work/tvm_fpga.cmds', 'fromtensorflow': False, 'weights': 'weights', 'datadir': 'work/weights_data', 'pngfile': 'graph.png', 'verbose': True, 'img_mean': [100], 'quantizecfg': 'work/tvm_quantization_params.json', 'calibration_size': 15, 'bitwidths': [16, 16, 16], 'img_raw_scale': 255.0, 'img_input_scale': 1.0, 'platform': 'alveo-u200', 'xclbin': '../overlaybins/alveo-u200/overlay_3.xclbin', 'scaleA': 1, 'scaleB': 1, 'PE': 0, 'batch_sz': 1, 'in_shape': (1, 4, 4)}
            #print(handles)
            print(config)
            #print(config2)
            #for key in config2.keys():
            #    if config[key] != config2[key]:
            #        print("Difference")
            #        print("c1: {}".format(config[key]))
            #        print("c2: {}".format(config2[key]))
            #time.sleep(10)
            self.fpga_rt = xdnn_lib.XDNNFPGAOp(handles, config)


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
