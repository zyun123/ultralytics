from ultralytics import YOLO
from PIL import Image
import cv2

img = cv2.imread("fs_data/test/1070.jpg")

model = YOLO("yolov8n.yaml").load("runs/detect/train3/weights/best.pt")
result = model(img,save = True)