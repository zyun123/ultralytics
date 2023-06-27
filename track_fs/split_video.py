import cv2

cap = cv2.VideoCapture("./track_fs/03.mp4")
count = 1
while cap.isOpened():
    
    ret,frame = cap.read()

    if not ret:
        break

    cv2.imshow("frame",frame)
    if count%10 == 0:
        cv2.imwrite(f"./fs_data/{count}.jpg",frame)
    
    key = cv2.waitKey(10)
    if key == 27:
        break
    count +=1

print("count:",count)   
cap.release()
cv2.destroyAllWindows()