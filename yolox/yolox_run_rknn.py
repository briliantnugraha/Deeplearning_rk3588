#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.
# python3 yolox_run.py --model yolox_tiny.onnx --image_path Images/

import argparse
import os
import time
from rknnlite.api import RKNNLite

import cv2
import numpy as np

import onnxruntime as ort

from yolox_utils import (COCO_CLASSES, 
                    preprocess, 
                    multiclass_nms_class_agnostic as multiclass_nms, 
                    demo_postprocess)
from yolox_vis import vis


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    return parser


def postprocess(output, input_shape, args, COCO_CLASSES):
    predictions = demo_postprocess(output[0], input_shape)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.01)
    return dets


if __name__ == '__main__':
    args = make_parser().parse_args()
    print('load parser finish.')

    input_shape = tuple(map(int, args.input_shape.split(',')))

    imglist = [os.path.join(args.image_path, ip) for ip in os.listdir(args.image_path) if '_out.' not in ip]
    imglist = imglist[2:]
    _rknnlite = RKNNLite(verbose=False, verbose_file='log.file')
    stat = _rknnlite.load_rknn(args.model)
    if stat != 0:
        print('Load rknn model failed!')
    else:
        print('Load rknn model success!')
    

    stat = _rknnlite.init_runtime(async_mode=True, core_mask=RKNNLite.NPU_CORE_AUTO)
    if stat != 0:
        print('Init runtime ENV failed!')
    else:
        print('Init runtime ENV success!')

    print('load model and img list finish.', len(imglist))
    timelist = []
    for _ in range(1):
        for i in range(len(imglist)):
            origin_img = cv2.imread(imglist[i])
            '''PREPROCESS'''
            start = time.time()
            img, ratio = preprocess(origin_img, input_shape, (0,1,2))
            print(i, 'preprocess finish.', img.shape, img.min(), img.max(), imglist[i])

            '''INFERENCE'''
            output = _rknnlite.inference(inputs=[img[None, :, :, :]])
            output[0] = output[0][...,0]
            start2 = time.time()
            print(i, 'inference finish. {:.3f}s'.format(start2-start))
            
            '''POSTPROCESS'''
            dets = postprocess(output, input_shape, args, COCO_CLASSES)
            start3 = time.time()
            timelist.append(start3-start)
            print(i, 'postprocess finish. {:.3f}s, total runtime: {:.3f}s'.format(start3-start2, timelist[-1]))

            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                                conf=args.score_thr, class_names=COCO_CLASSES)
            cv2.imwrite(imglist[i]+'_rknn.jpg', origin_img)

    
    print('Avg runtime: {:.3f}s'.format(np.mean(timelist[1:])))