from ultralytics import YOLO


model = YOLO("runs/pose/train19/weights/best.pt")

model.predict("/911G/dataest/middle_up_nei/coco/val",save = True,imgsz = 640,conf = 0.5)