from ultralytics import YOLO



model = YOLO("yolov8n-pose.pt")
results = model.train(data = 'coco_middle_up_nei.yaml',epochs = 100,imgsz = 640,fliplr = 0.0,batch = 16)