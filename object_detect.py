import cv2
import numpy as np
import imutils
import time

# Load the serialized caffe model from disk:
model = cv2.dnn.readNetFromCaffe("models/MobileNetSSD_deploy.prototxt", "models/MobileNetSSD_deploy.caffemodel")

# Load the object detection model
#model = cv2.dnn.readNetFromTensorflow('models/MobileNetSSD_deploy.prototxt', 
#                                      'models/MobileNetSSD_deploy.caffemodel')

# Initialize the video capture object
cap = cv2.VideoCapture(0) # for accessing the default camera
#cap = cv2.VideoCapture('path_to_video_file') # for accessing the video file

# used to record the time when we processed last frame 
prev_frame_time = 0

# used to record the time at which we processed current frame 
new_frame_time = 0

while True:
    # Read the frame
    ret, frame = cap.read()
    
    # Resize the frame
    frame = imutils.resize(frame, width=500)
    
    # Pass the frame to the object detection model
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (500, 500), 127.5)
    model.setInput(blob)
    detections = model.forward()
    
    # Loop through the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections
        if confidence > 0.5:
            # Get the object type
            class_id = int(detections[0, 0, i, 1])
            #class_name = classes[class_id]

            # Get the bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([500, 500, 500, 500])
            (startX, startY, endX, endY) = box.astype('int')
            
            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = '{}: {:.2f}%'.format(class_id, confidence * 100)
            cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # time when we finish processing for this frame 
    new_frame_time = time.time() 

    # fps will be number of frame processed in given time frame 
    # since their will be most of time error of 0.001 second 
    # we will be subtracting it to get more accurate result 
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
  
    # converting the fps into integer 
    fps = int(fps) 
  
    # converting the fps to string so that we can display it on frame 
    # by using putText function 
    fps = str(fps) 
  
    # putting the FPS count on the frame 
    cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (100, 255, 0), 3, cv2.LINE_AA) 
  
    # Show the frame
    cv2.imshow('Object Detection', frame)
    
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
        
# Release the video capture object and close all windows
cap.release()

