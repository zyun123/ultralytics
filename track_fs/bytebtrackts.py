from ultralytics import YOLO
import cv2
model = YOLO("yolov8n.pt")

results = model.track(source = "./track_fs/03.mp4",save = True,show = True)
# print(results)

# cap = cv2.VideoCapture("03.mp4")

# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# output_video = cv2.VideoWriter("output_video.mp4",fourcc,fps,(width,height))


# for result in results:
#     boxes = result.boxes
#     img = result.orig_img
#     for box in boxes:
#         xyxy = box.xyxy[0]
#         cv2.rectangle(img,(xyxy[0],xyxy[1]),(xyxy[2],xyxy[3]),(0,0,255),1)
    
#     output_video.write(img)

# cap.release()
# output_video.release()
