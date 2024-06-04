import copy
from ultralytics import YOLO
from PIL import Image
import cv2
import os
import numpy as np
from typing import DefaultDict
import json


models_map = {}
model = YOLO("runs/detect/train/weights/best.pt")

root_dir = "mydatasets/remote_train_all_jl_data/middle_up_nei/test"
new_dir = "mydatasets/hand_data/middle_up_nei/test"
os.makedirs(new_dir,exist_ok=True)

classes = {0: 'left_hand', 1: 'right_hand', 2: 'left_foot', 3: 'right_foot'}

colors = np.random.uniform(0, 255, size=(len(classes.values()), 3))


def generate_json_data(xyxy):
    xyxy = xyxy.tolist()
    

    shape = {
      "label": "hand",
      "points": [
            xyxy[:2],xyxy[2:]
      ],
      "group_id": None,
      "shape_type": "rectangle",
      "flags": {}
    }

    return shape
    

for file in os.listdir(root_dir):
    if file.endswith(".jpg"):
       
        print("file: " + file)
        img = cv2.imread(os.path.join(root_dir,file))
        results = model(img,save = False)
        results = results[0].cpu().numpy()
        boxes = results.boxes
        json_file = os.path.join(new_dir,file.replace(".jpg",".json"))

        data = {
            "version": "4.5.6", 
            "flags": {}, 
            "shapes":[],
            "imagePath": file, 
            "imageData": None, 
            "imageHeight": 720, 
            "imageWidth": 1280, 
            "flages": {}
        }

        for box in boxes:
            cls_id = int(box.cls)
            if cls_id in [0,1]:
                class_name = classes[cls_id]
                xyxy = box.xyxy.squeeze()
                xywh = box.xywh.squeeze()
                print(f"bbox w: {xywh[2]}, h: {xywh[3]}")
                label = f"{classes[cls_id]} {float(box.conf):.2f}"
                
                shape = generate_json_data(xyxy)
                # cv2.rectangle(img,(int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3])),colors[cls_id],1)
                # cv2.putText(img,label,(int(xyxy[0])-18,int(xyxy[1])-18),cv2.FONT_HERSHEY_SIMPLEX,0.5,colors[cls_id],2)

                data["shapes"].append(shape)

        with open(json_file,"w") as f:
            json.dump(data,f,indent = 4)




        cv2.imshow("res",img)
        cv2.waitKey(1)
        # cv2.imshow("res",crop_img)
        # cv2.waitKey(0)
        cv2.imwrite(f"{new_dir}/{file}",img)
cv2.destroyAllWindows()



            
        

