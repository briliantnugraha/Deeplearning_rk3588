import cv2
import numpy as np
import argparse
from rknnlite.api import RKNNLite


CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")


def get_argument():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        default="yolov10",
        help="yolov5 | yolov8 | yolov10",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="RK3588",
        help="Your rockchip type",
    )

    parser.add_argument(
        "--core",
        type=str,
        default=RKNNLite.NPU_CORE_0,
        help="RKNNLite.NPU_CORE_0 | RKNNLite.NPU_CORE_AUTO, etc",
    )

    parser.add_argument(
        "-m",
        "--modelpath",
        type=str,
        default="yolov10n.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default="Images/traffic.jpg",
        help="Path to your input image.",
    )
    parser.add_argument(
        "-s",
        "--score_thresh",
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
    parser.add_argument(
        "--disable_draw",
        default=False,
        action="store_true",
        help="Whether to not draw output results.",
    )
    args= parser.parse_known_args()[0]
    args.model_type = args.model_type.lower()
    
    print("ARGUMENTS: ", args)
    return args

def draw_result(image, boxes, scores, classes, show_detail=True):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    if show_detail:
        print("{:^12} {:^12}  {}".format('class', 'score', 'xmin, ymin, xmax, ymax'))
        print('-' * 50)
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        if show_detail:
            print("{:^12} {:^12.3f} [{:>4}, {:>4}, {:>4}, {:>4}]".format(CLASSES[cl], score, top, left, right, bottom))


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
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)



def get_prediction_info(output, score_thresh=0.3):
    predictions = np.squeeze(output[0]).T

    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    scores_filtered = scores > score_thresh
    predictions = predictions[scores_filtered, :]
    scores = scores[scores_filtered]
    boxes = predictions[:, :4]
    classes = np.argmax(predictions[:, 4:], axis=1)
    return boxes, scores, classes


def get_prediction_topk(output, score_thresh=0.3, topk=300):
    predictions = np.squeeze(output[0]).T

    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    scores_topk = np.argsort(-scores, axis=-1)[:topk]
    predictions = predictions[scores_topk, :]
    scores = scores[scores_topk]
    scores_filtered = scores > score_thresh
    # Get the class with the highest confidence
    boxes = predictions[scores_filtered, :4]
    boxes = np.maximum([0, 0, 0, 0], boxes)
    scores = scores[scores_filtered]
    classes = np.argmax(predictions[scores_filtered, 4:], axis=1)
    return boxes, scores, classes


def extract_boxes(boxes, im_inp, im_ori):
    # Scale boxes to original image dimensions
    boxes = rescale_boxes(boxes, im_inp, im_ori)

    # Convert boxes to xyxy format
    boxes = xywh2xyxy(boxes).astype(int)

    return boxes


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


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


def rescale_boxes(boxes, shapeinp=(640, 640), shapeori=(640, 640)):
    sinput = np.array(list(shapeinp[::-1])*2)
    sori = np.array(list(shapeori[::-1])*2)
    boxes = (boxes * sori / sinput).astype(np.float32)
    return boxes


def non_maximum_suppression(boxes, scores, iou_threshold=0.5):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(
            boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes
