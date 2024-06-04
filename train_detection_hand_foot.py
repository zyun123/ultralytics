from ultralytics import YOLO
from PIL import Image
import cv2



#-----------demo test bus.jpg-------------------
# model = YOLO("yolov8n.pt")
# img = Image.open("bus.jpg")
# result = model.predict(source = img,save = True)


params = {
    "batch":8,
    "epochs":500,
    "imgsz": 640,
    # "flipud":0.5,
    # "fliplr":0.5,
    "mosaic":1.0
}




#-----------------------train------------------------
model = YOLO("yolov8n.yaml").load("yolov8n.pt")
# model.train(data ="coco_hand_foot.yaml",**params)
model.train(data ="coco_fish.yaml",**params)