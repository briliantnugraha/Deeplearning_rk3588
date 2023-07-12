from rknnlite.api import RKNNLite
from typing import Any
import cv2
import numpy as np
import os
import argparse
import time
import logging as log
from yolov8_run import get_arguments, CLASS_NAME, COLORS, YOLOUtils, YOLOV8Runner


class YOLOV8RunnerRKNN(YOLOV8Runner):
    def __init__(self, path,
                core: int = RKNNLite.NPU_CORE_AUTO) -> None:
        log.info('Initialize YOLOV8RunnerRKNN start!')
        self.rknn_core = core
        self._load_model(path)
        self.get_input_details()
        self.set_default_threshold()
        log.info('Initialize YOLOV8RunnerRKNN finish!')

    def get_input_details(self):
        self.input_hw = [(480,480),(640, 640)][1]

    def _load_model(self, modelpath):
        self._rknnlite = RKNNLite(verbose=False, verbose_file='log.file')
        stat = self._rknnlite.load_rknn(modelpath)
        if stat != 0:
            log.info('Load rknn model failed!')
        log.info('Load rknn model success!')

        stat = self._rknnlite.init_runtime(async_mode=False, core_mask=self.rknn_core)
        if stat != 0:
            log.info('Init runtime ENV failed!')
        log.info('Init runtime ENV success!')

    def preprocess(self, image):
        self.img_wh = image.shape[:2][::-1]

        # Resize input image
        input_img = cv2.resize(image, tuple(self.input_hw[::-1]))
        return input_img

    def inference(self, x):
        print('x: ', x.shape, x.dtype)
        output = self._rknnlite.inference(inputs=[x], )
        print('output: ', type(output))
        output[0] = output[0][...,0]
        output[0][:,:4,:] = output[0][:,:4,:] * 32
        print('output shape: ', [o.shape for o in output])
        return output

   
if __name__ == '__main__':
    log.basicConfig(level=log.INFO)
    # get arguments
    args = get_arguments()

    # initialize model
    log.info('MAIN: Initalization start')
    yrunner = YOLOV8RunnerRKNN(args.model_path)
    yrunner.set_default_threshold(score_thresh=args.score_thresh,
                                  nms_thresh=args.nms_thresh)

    # do detection
    log.info('MAIN: Initalization success')
    
    imgpath = args.folder_path
    imglist = [os.path.join(imgpath, ip) for ip in os.listdir(imgpath) if '_out.' not in ip]
    timelist = []
    for _ in range(1):
        for i in range(len(imglist)):
            img_bgr = cv2.imread(imglist[i])
            start = time.time()
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            results = yrunner(img_rgb)
            timelist.append(time.time()-start)
            drawn_image = YOLOUtils.draw_output(img_bgr, *results)

            cv2.imwrite(imglist[i]+'_out.jpg', drawn_image)
            print(i, 'inference finish', imglist[i]+'_out.jpg', 'time: {:.3f}s'.format(timelist[-1]))
    print('avg time: ', np.mean(timelist[2:]))

