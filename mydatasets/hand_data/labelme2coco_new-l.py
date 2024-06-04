import json
import pycocotools
import numpy as np
import os
import glob


# def gen_annfile(ann_name):
#     info = {
#         "images":[],
#         "categories":[],
#         "annotations":[]
#     }
#     with open(f"{ann_name}.json","w") as f:
#         json.dump(info,f,indent = 4)




def labelme_to_coco(root_dir,json_files,ann_name):
    info = {
        "images":[],
        "categories":[],
        "annotations":[]
    }
    # gen_annfile(ann_name)

    for tplable in ["hand"]:
            cat_info = {
                "supercategory": tplable,
                "id": 1,
                "name": tplable
            }
            info["categories"].append(cat_info)
    num = 0
    for i,js_file in enumerate(json_files):
        print(js_file)
        with open(js_file,"r") as f:
            data = json.load(f)

        height = data["imageHeight"]
        width = data["imageWidth"]
        file_name = data["imagePath"].split("/")[-1]

        image_info = {
            "height":height,
            "width":width,
            "id":i,
            "file_name":file_name
        }
        info["images"].append(image_info)

        shapes = data["shapes"]
        labels = []
        for j,shape in enumerate(shapes):
            num += 1
            # label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            if shape_type == "rectangle":
                bbox = [points[0][0],points[0][1],
                        points[1][0] - points[0][0],
                        points[1][1] - points[0][1]]

                ann_info = {
                    "iscrowd": 0,
                    "image_id": i,
                    "category_id": 1,
                    "id": num,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                }
                info["annotations"].append(ann_info)


        

    
    with open(f"{root_dir}/{ann_name}.json","w") as f:
        json.dump(info,f,indent = 4)







if __name__ == "__main__":
    root_dir = "mydatasets/hand_data"
    img_dir = ["train","val"]

    for dir1 in img_dir:
        img_dir = os.path.join(root_dir,dir1)
        json_files = glob.glob(os.path.join(img_dir,"*.json"))


        labelme_to_coco(root_dir,json_files,dir1)