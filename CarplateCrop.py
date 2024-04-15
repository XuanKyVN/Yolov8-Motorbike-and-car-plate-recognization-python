import cv2
import os
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('C:/Users/Admin/PythonLession/yolo_dataset/best_carplate5.pt')

# Open the video file
video_path = "C:/Users/Admin/PythonLession/pic/carplate6.mp4"
cap = cv2.VideoCapture(video_path)
save_path = "C:/Users/Admin/PythonLession/CarPlate/Picture"
# Loop through the video frames
file_num = 0
unique_id = set()
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.track(frame, persist=True, conf=0.7)
        # print(results)
        if results[0].boxes.id != None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

            ids = results[0].boxes.id.cpu().numpy().astype(int)
            for box, id in zip(boxes, ids):
                # Check if the id is unique
                int_id = int(id)
                if int_id not in unique_id:
                    unique_id.add(int_id)

                    # Crop the image using the bounding box coordinates
                    cropped_img = frame[box[1]:box[3], box[0]:box[2]]
                    cropped_img = cv2.resize(cropped_img,(320,320))

                    # Save the cropped image with a unique filename
                    filename = f"liscense_{int_id}.jpg"
                    filepath = os.path.join(save_path, filename)
                    cv2.imwrite(filepath, cropped_img)

                # Draw the bounding box and id on the frame
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (85, 45, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(
                    frame,
                    f"Id {id}",
                    (box[0], box[1]),
                    0,
                    0.9,
                    [85, 45, 255],
                    2,
                    lineType=cv2.LINE_AA
                )
                cv2.imshow("Detected Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # Break the loop if 'q' is pressed
        #if cv2.waitKey(1) & 0xFF == ord("q"):
         #   break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()