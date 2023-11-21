from ultralytics import YOLO
from PIL import Image
import cv2


model = YOLO("yolov8n-pose.pt")

results = model.train(data = "coco-pose.yaml",epochs =100,imgsz = 640)