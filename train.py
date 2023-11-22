from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(data = "coco_hard_hat.yaml",epochs = 500,imgsz = 640)