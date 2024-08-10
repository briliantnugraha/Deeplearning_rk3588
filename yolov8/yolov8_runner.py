
import argparse
import os
import time
from rknnlite.api import RKNNLite

import cv2
import numpy as np
import abc
import time
from base_rknn import BaseRKNN
from base_utils import get_argument, draw_result, non_maximum_suppression, get_prediction_info, extract_boxes


class YOLOV8Runner(BaseRKNN):
    def __init__(self, 
                 modelpath: str, 
                 async_mode: bool = True, 
                 core: int = RKNNLite.NPU_CORE_0, #NPU_CORE_AUTO,
                 verbose: bool = False,
                 host_name="RK3588",
                 score_thresh = 0.3,
                 data_format=['nhwc']):
        super(YOLOV8Runner, self).__init__(modelpath, async_mode, core, verbose, host_name=host_name, data_format=data_format)
        self.score_thresh = score_thresh
        
    def do_preprocess(self, src=None):
        print('[YOLOV8] preprocess')
        src_prep =  np.ascontiguousarray(cv2.resize(src, (640, 640))[None])
        return src_prep
    
    def do_inference(self, src=None, src_prep=None):
        print('[YOLOV8] inference')
        outputs = self.rkengine.inference(inputs=[src_prep], data_format=self.data_format.copy())
        return outputs[0]
    
    def do_postprocess(self, src=None, src_prep=None, src_infer=None):
        print('[YOLOV8] postprocess')
        boxes, scores, classes = get_prediction_info(src_infer, score_thresh=0.3)
        boxes = extract_boxes(boxes, src_prep.shape[1:3], src.shape[:2])
        kp = non_maximum_suppression(boxes, scores)
        boxes, scores, class_ids = boxes[kp], scores[kp], classes[kp]
        return boxes, scores, class_ids


if __name__ == '__main__':
    args = get_argument()
    print("ARGUMENTS: ", args)
    
    yrunner = YOLOV8Runner(args.modelpath, core=args.core, host_name=args.host, score_thresh=args.score_thresh)

    img_ = cv2.imread(args.image_path)
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    
    for i in range(100):
        start = time.time()
        img = img_.copy()
        boxes, scores, ids = yrunner(img)[0]
        print(f"{i}. out: {boxes.shape}, {scores.shape}, {ids.shape}, time: {time.time() - start:.3f}s")
        # imdraw = img.copy()
    # draw_result(imdraw, boxes, scores, ids)
    # cv2.imwrite('result.jpg', imdraw[..., ::-1])