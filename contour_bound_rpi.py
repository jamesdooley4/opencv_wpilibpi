
from cscore import CameraServer
from ntcore import NetworkTableInstance

import cv2
import json
import numpy as np
import time

def main():
    with open('/boot/frc.json') as f:
        config = json.load(f)
    camera = config['cameras'][0]

    width = camera['width']
    height = camera['height']
    CameraServer.startAutomaticCapture()

    input_stream = CameraServer.getVideo()
    output_stream = CameraServer.putVideo('Processed', width, height)
        
    # Table for vision output information
    networkTableInstance = NetworkTableInstance.getDefault()
    vision_nt = networkTableInstance.getTable('Vision')

    # Allocating new images is very expensive, always try to preallocate
    img = np.zeros(shape=(240, 320, 3), dtype=np.uint8)

    # Wait for NetworkTables to start
    time.sleep(0.5)



    while True:
        start_time = time.time()

        frame_time, input_img = input_stream.grabFrame(img)
        output_img = np.copy(input_img)

        # Notify output of error and skip iteration
        if frame_time == 0:
            output_stream.notifyError(input_stream.getError())
            continue

        # Convert to HSV and threshold image
        hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
        #purple
        binary_img_purple = cv2.inRange(hsv_img, (120,100, 100), (130, 255, 255))
        #yellow
        binary_img_yellow = cv2.inRange(hsv_img, (18,100,100), (28, 255, 255))

        contour_list_purple, _ = cv2.findContours(binary_img_purple, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        contour_list_yellow, _ = cv2.findContours(binary_img_yellow, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        x_list = []
        y_list = []
        minContourSize = 100

        for contour in contour_list_purple:
            # Ignore small contours that could be because of noise/bad thresholding
            area = cv2.contourArea(contour)
            if area < minContourSize:
                continue

            cv2.drawContours(output_img, contour, -1, color = (255, 255, 255), thickness = -1)

            rect = cv2.minAreaRect(contour)
            center, size, angle = rect
            center = tuple([int(dim) for dim in center]) # Convert to int so we can draw

            # Draw rectangle and circle
            cv2.drawContours(output_img, [cv2.boxPoints(rect).astype(int)], -1, color = (0, 0, 255), thickness = 2)
            cv2.circle(output_img, center = center, radius = 3, color = (0, 0, 255), thickness = -1)
            cv2.putText(output_img, str(area), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

            x_list.append((center[0] - width / 2) / (width / 2))
            x_list.append((center[1] - width / 2) / (width / 2))

        vision_nt.putNumberArray('target_x_purple', x_list)
        vision_nt.putNumberArray('target_y_purple', y_list)

        x_list = []
        y_list = []

        for contour in contour_list_yellow:

            # Ignore small contours that could be because of noise/bad thresholding
            area = cv2.contourArea(contour)
            if area < minContourSize:
                continue

            cv2.drawContours(output_img, contour, -1, color = (255, 255, 255), thickness = -1)

            rect = cv2.minAreaRect(contour)
            center, size, angle = rect
            center = tuple([int(dim) for dim in center]) # Convert to int so we can draw

            # Draw rectangle and circle
            cv2.drawContours(output_img, [cv2.boxPoints(rect).astype(int)], -1, color = (255, 255, 0), thickness = 2)
            cv2.circle(output_img, center = center, radius = 3, color = (255, 255, 0), thickness = -1)
            cv2.putText(output_img, str(area), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
            

            x_list.append((center[0] - width / 2) / (width / 2))
            x_list.append((center[1] - width / 2) / (width / 2))

        vision_nt.putNumberArray('target_x_yellow', x_list)
        vision_nt.putNumberArray('target_y_yellow', y_list)

        processing_time = time.time() - start_time
        fps = 1 / processing_time
        cv2.putText(output_img, str(round(fps, 1)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        output_stream.putFrame(output_img)

main()