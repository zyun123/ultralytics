import torch
import cv2
import os
import torchvision
loaded_model = torch.jit.load("./hand_foot_model/left_hand_middle_up_nei.ts")
# img_path = "/911G/data/temp/20221229新加手托脚托新数据/20230311_最新修改/middle_up_nei/test_crop/left_foot/crop_m_up_nei_20221228151547322.jpg"
# src = cv2.imread(img_path)
# input_tensor = torch.tensor(src).permute(2,0,1).contiguous()
# input_tensor = torch.randn((3,720,1280))

root_dir = "yolo_crop/left_hand"
pred_dir = root_dir + "_pred"
os.makedirs(pred_dir,exist_ok=True)
with torch.no_grad():
    for file in os.listdir(root_dir):
        src = cv2.imread(os.path.join(root_dir,file))
        input_tensor = torch.tensor(src).permute(2,0,1).contiguous()
        output = loaded_model(input_tensor)
        # print("pred_boxes:/n",output[0])
        # print("pred_classes:/n",output[1])
        # print("pred_kp_heatmaps:/n",output[2])
        # print("keypoints:/n",output[3])
        # print("score:/n",output[4])
        # print("imgsize:/n",output[5]) #imgsize

        for kp in output[3].squeeze().cpu().numpy().tolist():
            x,y = int(kp[0]),int(kp[1])
            cv2.circle(src,(x,y),1,(0,0,255),2)
        
        cv2.imshow("res",src)
        cv2.waitKey(1)
        cv2.imwrite(f"{pred_dir}/{file}",src)
cv2.destroyAllWindows()