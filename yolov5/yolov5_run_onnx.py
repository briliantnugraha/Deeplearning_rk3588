import json
import os
import time
import numpy as np
from pathlib import Path

from rknnlite.api import RKNNLite
from yolov5_pp import yolov5_post_process

import cv2


def draw_output(image, boxes, classes, scores):
    hw = np.array(image.shape[:2][::-1]*2)/np.array([640,640]*2)
    print('image size: ', hw)
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = (box*hw).astype(int)
        print('class: {}, score: {}'.format(cfg['inference']['classes'][cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

        image = cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        image = cv2.putText(image, '{0} {1:.2f}'.format(cfg['inference']['classes'][cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
    return image

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )  # add border
    return im #, ratio, (dw, dh)



ROOT = Path(__file__).parent.absolute()
MODELS = str(ROOT) + "/models/"
CONFIG_FILE = str(ROOT) + "/config.json"
print('CONFIG_FILE: ', CONFIG_FILE, ROOT)

with open(CONFIG_FILE, 'r') as config_file:
    cfg = json.load(config_file)


class Yolov5():
    def __init__(
            self,
            core: int = RKNNLite.NPU_CORE_AUTO
    ):
        self._core = core
        #Check new model loaded
        
    def _load_model(self, model: str):
        self._rknnlite = RKNNLite(
            verbose=cfg["debug"]["verbose"],
            verbose_file=str(ROOT) + "/" + cfg["debug"]["verbose_file"]
        )
        ret = self._rknnlite.load_rknn(model)
        if ret != 0:
            print('%d. Export rknn model failed!')
            return ret
        print('%d. Init runtime environment')
        ret = self._rknnlite.init_runtime(
            async_mode=cfg["inference"]["async_mode"],
            core_mask = self._core
        )
        if ret != 0:
            print('%d. Init runtime environment failed!')
            return ret
        print('%s model loaded'%(model))
        return ret

    def preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(
            frame,
            (cfg["inference"]["net_size"], cfg["inference"]["net_size"])
        )
        return frame

    def inference(self, frame):
        frame_preprocess = self.preprocess(frame)
        outputs = self._rknnlite.inference(inputs=[frame_preprocess])
        return outputs


if __name__ == '__main__':
    model = Yolov5()
    modelpath = '../models/yolov5s.rknn'
    model._load_model(modelpath)
    
    basepath = '../Images'
    imglist = [os.path.join(basepath, bp) for bp in os.listdir(basepath) if '_out.' not in bp]
    for impath in imglist: 
        img = cv2.cvtColor(cv2.imread(impath), cv2.COLOR_BGR2RGB)
        timelist = []
        for _ in range(1):
            start = time.time()
            outputs_model = model.inference(img)
            start2 = time.time()
            boxes, classes, scores = yolov5_post_process(outputs_model)
            timelist.append([start2-start, time.time()-start2])
            print(_, 'infer-postprocess time: {:.3f}s-{:.3f}s'.format(*timelist[-1]), impath.split('/')[-1])
            print('bbox: ', len(boxes))
            print('classes: ', len(classes), classes[:10])
            print('scores: ', len(scores), scores[:10])
            print('='*30)
        image = draw_output(img, boxes, classes, scores)
        cv2.imwrite(impath+'_out.jpg', image[...,::-1])
    print('avg time: ', np.array(timelist[2:]).mean(0))