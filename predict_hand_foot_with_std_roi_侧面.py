import copy
from ultralytics import YOLO
from PIL import Image
import cv2
import os
import torch
import torchvision
import numpy as np
from typing import DefaultDict

"""
侧面视角
"""


models_map = {}
# model = YOLO("runs/detect/train/weights/best.pt")

# root_dir = "/911G/data/temp/20221229新加手托脚托新数据/20230311_最新修改/middle_up_nei/test"
root_dir = "/911G/data/新全经络数据/test_left_down_wai"

# classes = {0: 'left_hand', 1: 'right_hand', 2: 'left_foot', 3: 'right_foot'}
classes = {0: 'right_foot'}
for clsId , className in classes.items():
    models_map[clsId] = torch.jit.load(f"/911G/data/新全经络数据/20230410/{className}_left_down_wai.ts")

colors = np.random.uniform(0, 255, size=(len(classes.values()), 3))



for file in os.listdir(root_dir):
    if file.endswith(".jpg") and file.startswith("left_down_wai"):
    # if file.endswith(".jpg"):
        print(f"image file pred start {file}")
        # img = Image.open(os.path.join(root_dir,file))
        img = cv2.imread(os.path.join(root_dir,file),cv2.IMREAD_UNCHANGED)
        boxes = {0:[60,180]}   #------------------固定roi

        for cls_id, box in boxes.items():
            
            class_name = classes[cls_id]
            # xyxy = box.xyxy.squeeze()
            label = f"{classes[cls_id]} {float(1.0):.2f}"
            
            # center_p = [round((xyxy[0] + xyxy[2])/2),round((xyxy[1] + xyxy[3])/2)]
            crop_img = copy.deepcopy(img[box[1]:box[1]+320,box[0]:box[0]+320:])
            cv2.imwrite(f"yolo_crop/left_down_wai_right_foot/{file}_crop.jpg",crop_img)
            crop_img = cv2.resize(crop_img,(256,256),interpolation =cv2.INTER_LINEAR)

            scale = 256/320
            # cv2.imshow("crop_img",crop_img)
            # cv2.waitKey(1)

            # cv2.rectangle(img,(round(xyxy[0]),round(xyxy[1])),(round(xyxy[2]),round(xyxy[3])),colors[cls_id],1)
            # cv2.putText(img,label,(round(xyxy[0])-10,round(xyxy[1])-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,colors[cls_id],2)
            cv2.rectangle(img,(box[0],box[1]),(box[0]+320,box[1]+320),colors[cls_id],1)
            cv2.putText(img,label,(box[0]-10,box[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,colors[cls_id],2)


            input_tensor = torch.tensor(crop_img).permute(2,0,1).contiguous()
            with torch.no_grad():
                output = models_map[cls_id](input_tensor)
                for kp in output[3].cpu().numpy().tolist()[0]:
                    x,y = int(kp[0]/scale)+box[0],int(kp[1]/scale)+box[1]
                    cv2.circle(img,(x,y),2,(0,0,255),2)
        
        cv2.imshow("res",img)
        cv2.waitKey(1)
        cv2.imwrite(f"results_left_down_wai/{file}",img)
cv2.destroyAllWindows()



            
        

