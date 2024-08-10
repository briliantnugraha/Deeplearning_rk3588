
import argparse
import os
import time
from rknnlite.api import RKNNLite

import cv2
import numpy as np
import abc
import time
from base_rknn import BaseRKNN
from base_utils import get_argument, draw_result, get_prediction_topk, rescale_boxes, non_maximum_suppression, get_prediction_info


class YOLOV10Runner(BaseRKNN):
    def __init__(self, 
                 modelpath: str, 
                 async_mode: bool = False, 
                 core: int = RKNNLite.NPU_CORE_0, #NPU_CORE_AUTO,
                 verbose: bool = False,
                 host_name="RK3588",
                 score_thresh = 0.3,
                 data_format=['nhwc']):
        super(YOLOV10Runner, self).__init__(modelpath, async_mode, core, verbose, host_name=host_name, data_format=data_format)
        self.score_thresh = score_thresh
        self.topk = 300
        
    def do_preprocess(self, src=None):
        print('[YOLOV10] preprocess')
        src_prep =  cv2.resize(src, (640, 640))[None]
        return src_prep
    
    def do_inference(self, src=None, src_prep=None):
        print('[YOLOV10] inference')
        outputs = self.rkengine.inference(inputs=[src_prep], data_format=self.data_format.copy())
        return outputs[0]
    
    def do_postprocess(self, src=None, src_prep=None, src_infer=None):
        print('[YOLOV10] postprocess')
        boxes, scores, classes = get_prediction_topk(src_infer, score_thresh=self.score_thresh, topk=self.topk)
        boxes = rescale_boxes(boxes, src_prep.shape[1:3], src.shape[:2])
        kp = non_maximum_suppression(boxes, scores)
        boxes, scores, class_ids = boxes[kp], scores[kp], classes[kp]
        return boxes, scores, class_ids


if __name__ == '__main__':
    args = get_argument()
    
    yrunner = YOLOV10Runner(args.modelpath, core=args.core, host_name=args.host, score_thresh=args.score_thresh)

    img_ = cv2.imread(args.image_path)
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    
    for i in range(100):
        img = img_.copy()
        start = time.time()
        out = yrunner(img)[0]
        print(f"{i}. out: {out.shape}, time: {time.time() - start:.3f}s")
    draw_result(img, out[0], out[1], out[2])
    
    cv2.imwrite('result_yolov10.jpg', img)