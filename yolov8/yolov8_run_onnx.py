# this code is heavily derived from: https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection/
from typing import Any
try:
    import onnxruntime as ort
except:
    print('onnxruntime does not exists, proceed without import ORT')
import cv2
import numpy as np
import os
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-st", "--score_thresh", default=0.25, type=float, help="score threshold for yolov8",)
    parser.add_argument("-nt", "--nms_thresh", default=0.3, type=float, help="nms threshold for yolov8",)
    parser.add_argument("-mp", "--model_path", default="yolov8s.onnx", type=str, help="model path for yolov8",)
    parser.add_argument("-fp", "--folder_path", default="Images", type=str, help="folder path that contains images for yolov8 to detect",)
    return parser.parse_args()


CLASS_NAME = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
COLORS = rng.uniform(0, 255, size=(len(CLASS_NAME), 3))


class YOLOUtils:
    @staticmethod
    def draw_output(img, boxes, scores, class_ids):
        img_height, img_width = img.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)
        for box, score, class_id in zip(boxes, scores, class_ids):
            color = COLORS[class_id]

            x1, y1, x2, y2 = box.astype(int)

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = CLASS_NAME[class_id]
            caption = f'{label} {int(score * 100)}%'
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(img, (x1, y1),
                        (x1 + tw, y1 - th), color, -1)
            cv2.putText(img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)
        return img 

    def get_input_details(self):
        model_inputs = self.model.get_inputs()
        self.input_names = [m.name for m in self.model.get_inputs()]
        self.input_hw = model_inputs[0].shape[2:]

    def get_output_details(self):
        model_outputs = self.model.get_outputs()
        self.output_names = [m.name for m in self.model.get_outputs()]

    def set_default_threshold(self, score_thresh=0.5, nms_thresh=0.3):
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

    def get_prediction_info(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        scores_filtered = scores > self.score_thresh
        predictions = predictions[scores_filtered, :]
        scores = scores[scores_filtered]
        return predictions, scores

    @staticmethod
    def compute_iou(box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(
            0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou

    @staticmethod
    def non_maximum_suppression(boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = YOLOUtils.compute_iou(
                boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]

            # print(keep_indices.shape, sorted_indices.shape)
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes

    @staticmethod
    def xywh2xyxy(x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = YOLOUtils.xywh2xyxy(boxes)

        return boxes
    
    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array(self.input_hw[::-1]*2)
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array(self.img_wh*2)
        return boxes


class YOLOV8Runner(YOLOUtils):
    def __init__(self, path) -> None:
        self.model = ort.InferenceSession(
            path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.get_input_details()
        self.get_output_details()
        self.set_default_threshold()

    def preprocess(self, image):
        self.img_wh = image.shape[:2][::-1]

        # Resize input image
        input_img = cv2.resize(
            image, tuple(self.input_hw[::-1]))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0

        # set to B C H W format (pytorch input format)
        input_tensor = input_img.transpose(2, 0, 1)[None].astype(np.float32)
        return input_tensor

    def inference(self, x):
        return self.model.run(None, {self.input_names[0]: x})

    def postprocess(self, output):
        predictions, scores = self.get_prediction_info(output)

        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([])

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = self.non_maximum_suppression(
            boxes, scores, self.nms_thresh)

        return boxes[indices], scores[indices], class_ids[indices]

    def __call__(self, x) -> list:
        inp = self.preprocess(x)
        infer_out = self.inference(inp)
        final_out = self.postprocess(infer_out)
        return final_out
    

if __name__ == '__main__':
    # get arguments
    args = get_arguments()

    # initialize model
    yrunner = YOLOV8Runner(args.model_path)
    yrunner.set_default_threshold(score_thresh=args.score_thresh,
                                  nms_thresh=args.nms_thresh)
    
    # prepare img path list
    imgpath = args.folder_path
    imglist = [os.path.join(imgpath, ip) for ip in os.listdir(imgpath) if '_out.' not in ip]

    # do inference for each image
    for i in range(len(imglist)):
        img_bgr = cv2.imread(imglist[i])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        results = yrunner(img_rgb)
        drawn_image = YOLOUtils.draw_output(img_bgr, *results)
        cv2.imwrite(imglist[i].replace('.', '_out.'), drawn_image)
        print(i, 'inference finish', imglist[i].replace('.', '_out.'))