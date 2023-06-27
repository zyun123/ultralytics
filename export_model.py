from ultralytics import YOLO
# model = YOLO("runs/detect/train/weights/best.pt")
model = YOLO("yolov8n.pt")
model.export(format = "torchscript",imgsz = 640,opset = 12)