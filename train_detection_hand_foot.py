from ultralytics import YOLO
from PIL import Image
import cv2



#-----------demo test bus.jpg-------------------
# model = YOLO("yolov8n.pt")
# img = Image.open("bus.jpg")
# result = model.predict(source = img,save = True)



#-----------------------train------------------------
model = YOLO("yolov8n.yaml").load("yolov8n.pt")
model.train(data ="coco_hand_foot.yaml",epochs = 100,imgsz = 640)