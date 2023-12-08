
#from cscore import CameraServer
#from networktables import NetworkTables

import fractions
from lib2to3.fixes.fix_next import bind_warning
from tkinter.messagebox import showerror
import cv2
import json
import imutils
import numpy as np
import time

def main():
    cap = cv2.VideoCapture(1) # for accessing the default camera

    width = 500;
    n_clusters = 4

    # Allocating new images is very expensive, always try to preallocate
    img = np.zeros(shape=(240, 320, 3), dtype=np.uint8)

    # Wait for NetworkTables to start
    #time.sleep(0.5)

    showColorCluster = False

    while True:
        start_time = time.time()

        #frame_time, input_img = input_stream.grabFrame(img)
        frame_time, input_img = cap.read()
        # Resize the frame
        input_img = imutils.resize(input_img, width=500)
        output_img = np.copy(input_img)
        #output_img = np.multiply(output_img, np.array([1,1.5,1])).astype(np.uint8)


        # Notify output of error and skip iteration
        if frame_time == 0:
            #output_stream.notifyError(input_stream.getError())
            continue

        # Convert to HSV and threshold image
        hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
        #purple
        binary_img_purple = cv2.inRange(hsv_img, (120,100, 100), (130, 255, 255))
        #yellow
        binary_img_yellow = cv2.inRange(hsv_img, (18,100,100), (28, 255, 255))
        #output_img = binary_img

        if showColorCluster:
            #extract hue (0) column
            #data =  hsv_img.reshape(-1, 3)[:, 0]
            data =  hsv_img.reshape(-1, 3)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            flags = cv2.KMEANS_RANDOM_CENTERS
            compactness, labels, centers = cv2.kmeans(data.astype(np.float32), n_clusters, None, criteria, 10, flags)

            cluster_sizes = np.bincount(labels.flatten())

            palette = np.empty((0, 3)).astype(np.uint8)
            for cluster_idx in np.argsort(-cluster_sizes):
                #palette = np.append(palette, np.array([[centers[cluster_idx].astype(int)[0],230,230]]), axis=0)
                palette = np.append(palette, np.array([centers[cluster_idx].astype(np.uint8)]), axis=0)
                
                #palette.append(np.full((hsv_img.shape[0], hsv_img.shape[1], 3), fill_value=centers[cluster_idx].astype(int), dtype=np.uint8))
            palette = np.reshape(palette,(1,palette.shape[0], palette.shape[1]))
            #palette = np.hstack(palette)


        contour_list_purple, _ = cv2.findContours(binary_img_purple, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        contour_list_yellow, _ = cv2.findContours(binary_img_yellow, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        x_list = []
        y_list = []
        
        minContourSize = 800

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
        #vision_nt.putNumberArray('target_x_purple', x_list)
        #vision_nt.putNumberArray('target_y_purple', y_list)

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

        #vision_nt.putNumberArray('target_x_yellow', x_list)
        #vision_nt.putNumberArray('target_y_yellow', y_list)

        if showColorCluster:
            sf = hsv_img.shape[1] / palette.shape[1]
            palette_bgr = cv2.cvtColor(palette, cv2.COLOR_HSV2BGR)
            palette_block = cv2.resize(palette_bgr, (0, 0), fx=sf, fy=sf, interpolation=cv2.INTER_NEAREST)
            output_img = np.vstack([output_img, palette_block])
            cv2.putText(output_img, "H:" + str(palette[0][0][0]), (80, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (00, 0, 0))
            cv2.putText(output_img, "S:" + str(palette[0][0][1]), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            cv2.putText(output_img, "V:" + str(palette[0][0][2]), (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

        processing_time = time.time() - start_time
        fps = 1 / processing_time
        cv2.putText(output_img, str(round(fps, 1)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))


        #output_stream.putFrame(output_img)
        # Show the frame
        cv2.imshow('Object Detection', output_img)
    
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


main()