from pycocotools.coco import COCO
import numpy as np
import os
import tqdm
import shutil
import argparse

def default_argparse():
    parser = argparse.ArgumentParser(description = "coco2yolo argument")
    parser.add_argument("--dataset-type",default = "val",help = "train dir or test dir")  
    parser.add_argument("--copy-images",action = "store_true",help = "copy images to yolo dir  yes or not") 
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # coco_dir = "mydatasets/hand_data"
    # yolo_dir = "mydatasets/hand_data_yolo"

    coco_dir = "hand_data"
    yolo_dir = "hand_data_yolo"


    args = default_argparse()
    # args.copy_images = True
    # args.dataset_type = "val"

    dataset_name = args.dataset_type
    save_labels_dir = f"{yolo_dir}/{dataset_name}/labels"  #save images dir
    save_images_dir = f"{yolo_dir}/{dataset_name}/images"  #save labels dir
    os.makedirs(save_labels_dir,exist_ok = True)

    save_base_dir = yolo_dir  #yolo base dir

    ann_file = f"{coco_dir}/{dataset_name}.json"   #coco annotations
    file_names = []
    

    #copy images to yolo images dir 
    if args.copy_images:
        os.makedirs(save_images_dir,exist_ok = True)

        for file in os.listdir(f"{coco_dir}/{dataset_name}"):
            if file.endswith(".jpg"):
                ori_file = os.path.join(f"{coco_dir}/{dataset_name}",file)
                new_file = os.path.join(save_images_dir,file)
                shutil.copy(ori_file,new_file)



    anns_data = COCO(annotation_file = ann_file)
    catIds = anns_data.getCatIds()
    categories = anns_data.loadCats(catIds)
    categories.sort(key = lambda x: x['id'])
    classes = {}
    coco_labels = {}
    coco_labels_inverse = {}
    for c in categories:
        coco_labels[len(classes)] = c['id']
        coco_labels_inverse[c['id']] = len(classes)
        classes[c['name']] = len(classes)




    img_ids = anns_data.getImgIds()
    for index , img_id in tqdm.tqdm(enumerate(img_ids),desc = 'change .json to .txt file'):
        img_info = anns_data.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        # file_names.append(os.getcwd() +f"/{yolo_dir}/{dataset_type}/images/"+file_name + "\n")
        file_names.append(os.path.join(os.getcwd(),save_images_dir,file_name)+"\n")

        height = img_info["height"]
        width = img_info["width"]

        ann_id = anns_data.getAnnIds(img_id)
        if len(ann_id) == 0:
            continue
        save_txt_file = os.path.join(save_labels_dir,file_name.replace("jpg","txt"))
        with open(save_txt_file,'w') as f:
            anns = anns_data.loadAnns(ann_id)
            lines = ''
            for ann in anns:
                box = ann["bbox"]
                if box[2] < 1 or box[3] <1:
                    continue
                x_center = round((box[0] + box[2] / 2) / width, 3)
                y_center = round((box[1] + box[3] / 2) / height, 3)
                norm_width = round(box[2] / width,3)
                norm_height = round(box[3] / height,3)
                label = coco_labels_inverse[ann['category_id']]
                lines = lines + str(label) + " " + str(x_center) + " " + str(y_center) + " " + str(norm_width) + " " + str(norm_height)
                if "keypoints" in ann:
                    kps = ann["keypoints"]
                    for i, kp in enumerate(kps):
                        if i%3 == 0:
                            lines += " " + str(round(kp/width,3))
                        elif i%3 == 1:
                            lines += " " + str(round(kp/height,3))
                        else:
                            lines += " " + str(kp)
                lines += "\n"
            f.writelines(lines)
    with open(os.path.join(save_base_dir,f"{dataset_name}.txt"),"w") as f:
        # for file in os.listdir(save_labels_dir):
        #     label_path = os.path.join(save_labels_dir,file)
        #     image_path = label_path.replace("labels","images").replace(".txt",".jpg")
        #     f.write(image_path + "\n")
        for file in file_names:
            f.write(file)


    print("finish")
                
                
			
             


                



          

