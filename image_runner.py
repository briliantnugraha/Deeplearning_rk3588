from yolov5.yolov5_runner import YOLOV5Runner
from yolov8.yolov8_runner import YOLOV8Runner
from yolov10.yolov10_runner import YOLOV10Runner
from base_utils import get_argument, draw_result
import argparse
import cv2
import time
import numpy as np


if __name__ == '__main__':
    args = get_argument()
    
    if args.model_type == 'yolov5':
        yrunner = YOLOV5Runner(args.modelpath, core=args.core, host_name=args.host, score_thresh=args.score_thresh)
    elif args.model_type == 'yolov8':
        yrunner = YOLOV8Runner(args.modelpath, core=args.core, host_name=args.host, score_thresh=args.score_thresh)
    elif args.model_type == 'yolov10':
        yrunner = YOLOV10Runner(args.modelpath, core=args.core, host_name=args.host, score_thresh=args.score_thresh)
    else:
        raise NotImplementedError
    
    # img_ = cv2.imread(args.image_path)
    # img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    
    for i in range(100):
        randpath = np.random.choice(['Images/im1.jpg', 'Images/im2.png', 'Images/traffic.jpg', 'Images/bus.jpg', 'Images/truck.jpg'])
        # randpath = 'Images/bus.jpg'
        img_ = cv2.imread(randpath)
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        start = time.time()
        img = img_.copy()
        boxes, scores, ids = yrunner(img)[0]
        print(f"{i}. out: {boxes.shape}, {scores.shape}, {ids.shape}, time: {time.time() - start:.3f}s")
        if not args.disable_draw:
            imdraw = img.copy()
            draw_result(imdraw, boxes, scores, ids, show_detail=False)
            cv2.imwrite('result.jpg', imdraw[..., ::-1])
        # break
    