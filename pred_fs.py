from ultralytics import YOLO
from PIL import Image
import cv2

img = cv2.imread("fs_data/test/1070.jpg")

model = YOLO("runs/detect/train3/weights/last.pt")
# result = model(img,save = True)


#############track

# results = model.track(source ="track_fs/03.mp4",show = True,save = True,tracker = "bytetrack.yaml")
results = model(source = "track_fs/03.mp4",save = True)