import cv2
from ultralytics import YOLO
import numpy as np

# load yolo pretrain Model  COCO128.yaml
model = YOLO('yolov8n.pt')
# open video
cap = cv2.VideoCapture("pic/vehicle.mp4")
# loop through Video frame
# read a frame from video
while cap.isOpened():
    success, frame = cap.read()
x1 = 297
x2 = 615
y1 = 49
y2 = 188
frame_cut = frame[y1:y2, x1:x2]  # (Height, Width)
# draw Zone detect
start_point = (x1, y1)
end_point = (x2, y2)
color = (255, 255, 0)
thickness = 2
frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
if success:
    ''' run YOLO V8 to tract '''
results = model.tract(frame_cut, persist=True)
for result in results:
    if result:
        boxes = result.boxes.numpy()  # Boxes object for bboz output
        for box in boxes:
            print("class", box.cls)
            x11 = int(box.xyxy[0][0]) + x1
            y11 = int(box.xyxy[0][1]) + y1
            x21 = int(box.xyxy[0][2]) + x1
            y21 = int(box.xyxy[0][3]) + y1
            # draw rectangle
            frame = cv2.rectangle(frame, (x11, y11), (x21, y21), color, thickness)
            # draw text
            location = (x11, y11 - 5)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            if box.id:
                text = "ID: " + str(int(box.id[0]))
                image = cv2.putText(frame, text, location, font, fontScale, color(0, 0, 255), thickness, cv2.LINE_AA)
        # display the frame
        cv2.imshow("YOLOV8 TRACKING", frame)
        # Break the loop if "q" is press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        else:
            break
            # release the video captuare object and close the display
    cap.release()
    cv2.destroyWindow()
