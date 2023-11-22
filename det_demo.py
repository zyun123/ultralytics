from ultralytics import YOLO

model = YOLO("runs/detect/train5/weights/best.pt")
# reuslts = model("/911G/data/hard_hat/trt_test")
model.predict("/911G/data/hard_hat/train",save = True, imgsz = 640,conf = 0.5)