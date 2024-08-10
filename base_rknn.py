import argparse
import os
import time
from rknnlite.api import RKNNLite

import cv2
import numpy as np
import abc


class BaseRKNN(metaclass=abc.ABCMeta):
    def __init__(self, 
                 modelpath: str, 
                 async_mode: bool = False, 
                 core: int = RKNNLite.NPU_CORE_AUTO,
                 verbose: bool = True,
                 host_name="RK3588",
                 verbose_file: str = "log.txt",
                 data_format: list[str] = ['nhwc']):
        self.rkengine = RKNNLite(verbose=False, verbose_file=verbose_file)
        self.data_format = data_format
        self.host_name = host_name

        model_stat = self.rkengine.load_rknn(modelpath)
        if model_stat != 0:
            raise ValueError(f"Load rknn model failed!, input: {modelpath}")
        print('[RKLoader] Load rknn model success!')

        if host_name in ['RK3576', 'RK3588']:
            # For RK3576 / RK3588, specify which NPU core the model runs on through the core_mask parameter.
            stat = self.rkengine.init_runtime(async_mode=False, core_mask=core)
            print(f"[RKLoader] LOAD {host_name}")    
        else:
            stat = self.rkengine.init_runtime()
            print(f"[RKLoader] LOAD {host_name}, no args init")    
        if stat != 0:
            raise NotImplementedError('Init runtime ENV failed!')
        print('[RKLoader] Init runtime ENV success!')
        
    def do_preprocess(self, src=None):
        raise NotImplementedError
    
    def do_inference(self, src=None, src_prep=None):
        raise NotImplementedError
    
    def do_postprocess(self, src=None, src_prep=None, src_infer=None):
        raise NotImplementedError
    
    def __call__(self, src):
        src_prep = self.do_preprocess(src)
        src_infer = self.do_inference(src, src_prep)
        src_out = self.do_postprocess(src, src_prep, src_infer)
        return src_out, src_infer, src_prep
