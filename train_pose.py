from ultralytics import YOLO



model = YOLO("yolov8n-pose.pt")
results = model.train(data = 'coco8-pose.yaml',epochs = 1000,imgsz = 640)