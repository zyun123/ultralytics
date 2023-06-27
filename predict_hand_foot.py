import copy
from ultralytics import YOLO
from PIL import Image
import cv2
import os
import torch
import torchvision
import numpy as np
from typing import DefaultDict

models_map = {}
model = YOLO("runs/detect/train/weights/best.pt")

# root_dir = "/911G/data/temp/20221229新加手托脚托新数据/20230311_最新修改/middle_up_nei/test"
root_dir = "/911G/data/cure_images/dynamic_up_nei/ori_img_01"

classes = {0: 'left_hand', 1: 'right_hand', 2: 'left_foot', 3: 'right_foot'}



for clsId , className in classes.items():
    models_map[clsId] = torch.jit.load(f"hand_foot_model/{className}_middle_up_nei.ts")

colors = np.random.uniform(0, 255, size=(len(classes.values()), 3))



for file in os.listdir(root_dir):
    if file.endswith(".jpg"):
        # img = Image.open(os.path.join(root_dir,file))
        print("file: " + file)
        img = cv2.imread(os.path.join(root_dir,file))
        results = model(img,save = False)
        results = results[0].cpu().numpy()
        boxes = results.boxes

        for box in boxes:
            cls_id = int(box.cls)
            class_name = classes[cls_id]
            xyxy = box.xyxy.squeeze()
            label = f"{classes[cls_id]} {float(box.conf):.2f}"
            
            center_p = [round((xyxy[0] + xyxy[2])/2),round((xyxy[1] + xyxy[3])/2)]
            # crop_img = copy.deepcopy(img[center_p[1]-100:center_p[1]+100,center_p[0]-128:center_p[0]+128:])
            crop_img = np.asarray(img[center_p[1]-100:center_p[1]+100,center_p[0]-100:center_p[0]+100:],dtype = np.uint8)
            crop_img = cv2.resize(crop_img,(256,256),interpolation =cv2.INTER_LINEAR)
            
            scale = 256/200

            # crop_dir=  f"yolo_crop/{classes[cls_id]}"
            # os.makedirs(crop_dir,exist_ok=True)
            # cv2.imwrite(f"{crop_dir}/{file}",crop_img)
            # cv2.rectangle(img,(round(xyxy[0]),round(xyxy[1])),(round(xyxy[2]),round(xyxy[3])),colors[cls_id],1)
            # cv2.putText(img,label,(round(xyxy[0])-10,round(xyxy[1])-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,colors[cls_id],2)
            


            input_tensor = torch.tensor(crop_img).permute(2,0,1).contiguous()
            with torch.no_grad():
                output = models_map[cls_id](input_tensor)
                for kp in output[3].cpu().numpy().tolist()[0]:
                    x,y = int((kp[0])/scale)+center_p[0]-100,int((kp[1])/scale)+center_p[1]-100
                    cv2.circle(img,(x,y),2,(0,0,255),2)
                    # x,y = int(kp[0]),int(kp[1])
                    # cv2.circle(crop_img,(x,y),2,(0,0,255),2)
            # cv2.imshow("res",crop_img)
            # cv2.waitKey(0)
            cv2.rectangle(img,(center_p[0]-100,center_p[1]-100),(center_p[0]+100,center_p[1]+100),colors[cls_id],1)
            cv2.putText(img,label,(center_p[0]-110,center_p[1]-110),cv2.FONT_HERSHEY_SIMPLEX,0.5,colors[cls_id],2)

        cv2.imshow("res",img)
        cv2.waitKey(0)
        # cv2.imshow("res",crop_img)
        # cv2.waitKey(0)
        cv2.imwrite(f"results/{file}",img)
cv2.destroyAllWindows()



            
        

