# Deeplearning with Rockchip

![](orangepi.jpg)

```
Hi,
In here, I am planning to share the journey of using rknn model with Rockchip SBC.
In my case, I use 4GB (cheapest RK3588s on the market) version of Orange pi 5.
I will consider to add rknn-toolkit 1.5 installation (which is very simple TBH).
```

## YOLOV5 - YOLOX

---

### How To Use

1. Clone the repository.
2. Do Inference:
  * YOLOV5 - Do inference with Images folder's contents `python3 yolov5_run.py`
  * YOLOX - Do inference with Images folder's contents (Download ONNX Model in [here](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx))
```
cd yolox
ONNXRuntime -> python3 yolox_run.py --model ../models/yolox_s.rknn --image_path Images/ --output_dir Images/ --input_shape 640,640
RKNN -> python3 yolox_run_rknn.py --model ../models/yolox_s.rknn --image_path Images/ --output_dir Images/ --input_shape 640,640
```
3. You can also add customized images, put it on the "Images" folder, and run step 2 again (image outputs will not be put into the inference pipeline, so no need to delete it).
4. Do inference with webcam
```
YOLOV5 -> python3 yolov5_run_webcam.py
YOLOX -> TBD
YOLO-V8 -> TBD
```

## Demo and Screenshot Results
<video controls="true" allowfullscreen="true">
    <source src="./demo_video.mp4" type="video/mp4">
  </video>

|Original|Detection|
|---|---|
| ![](Images/traffic.jpg) | ![](Images/traffic_out.jpg) |
| ![](Images/bus.jpg)| ![](Images/bus_out.jpg) |  |

---

## Acknowledgements

- https://github.com/Applied-Deep-Learning-Lab/Yolov5_RK3588
- YOLOV5 - Ultralytics