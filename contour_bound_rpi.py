
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

    n_clusters = 4
    clusterInterval = 10
    currentInterval = 0

    width = 320 # camera['width']
    height = 240 # camera['height']
    CameraServer.startAutomaticCapture()

    input_stream = CameraServer.getVideo()
    output_stream = CameraServer.putVideo('Processed', width, height)
        
    # Table for vision output information
    networkTableInstance = NetworkTableInstance.getDefault()
    networkTableInstance.startServer("/home/lvuser/networktables.ini")
    vision_nt = networkTableInstance.getTable('Vision')

    # Allocating new images is very expensive, always try to preallocate
    img = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    # wait for the NT server to actually start
    for i in range(100):
        if (
            networkTableInstance.getNetworkMode()
            & NetworkTableInstance.NetworkMode.kNetModeStarting
        ) == 0:
            break
        # real sleep since we're waiting for the server, not simulated sleep
        time.sleep(0.010)
    #else:
        #reportErrorInternal(
        #    "timed out while waiting for NT server to start", isWarning=True
        #)

    showColorCluster = False
    palette = np.empty((1,4,3)).astype(np.uint8)

    while True:
        start_time = time.time()
        currentInterval = currentInterval + 1

        frame_time, input_img = input_stream.grabFrame(img)
        output_img = np.copy(input_img)

        vision_nt.putNumberArray('inputImg_shape', input_img.shape)
        width = input_img.shape[1]
        height = input_img.shape[0]

        # Notify output of error and skip iteration
        if frame_time == 0:
            output_stream.notifyError(input_stream.getError())
            continue

        # Convert to HSV and threshold image
        hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
        #purple
        binary_img_purple = cv2.inRange(hsv_img, (120,100, 100), (152, 255, 255))
        #yellow
        binary_img_yellow = cv2.inRange(hsv_img, (18,100,100), (28, 255, 255))

        if showColorCluster and currentInterval > clusterInterval:
            #extract hue (0) column
            #data =  hsv_img.reshape(-1, 3)[:, 0]
            data =  hsv_img.reshape(-1, 3)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            flags = cv2.KMEANS_RANDOM_CENTERS
            compactness, labels, centers = cv2.kmeans(data.astype(np.float32), n_clusters, None, criteria, 10, flags)

            cluster_sizes = np.bincount(labels.flatten())

            palette = np.empty((0, 3)).astype(np.uint8)
            for cluster_idx in np.argsort(-cluster_sizes):
                palette = np.append(palette, np.array([centers[cluster_idx].astype(np.uint8)]), axis=0)
            palette = np.reshape(palette,(1,palette.shape[0], palette.shape[1]))


        contour_list_purple, _ = cv2.findContours(binary_img_purple, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        contour_list_yellow, _ = cv2.findContours(binary_img_yellow, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        x_list = []
        y_list = []
        area_list = []
        minContourSize = 100
        rawx_list = []
        rawy_list = []

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

            rawx_list.append(center[0])
            rawy_list.append(center[1])
            x_list.append((center[0] - width / 2) / (width / 2))
            y_list.append((center[1] - height / 2) / (height / 2))
            area_list.append(area)

        vision_nt.putNumberArray('target_rawx_purple', rawx_list)
        vision_nt.putNumberArray('target_rawy_purple', rawy_list)
        vision_nt.putNumberArray('target_x_purple', x_list)
        vision_nt.putNumberArray('target_y_purple', y_list)
        vision_nt.putNumberArray('target_area_purple', area_list)

        x_list = []
        y_list = []
        area_list = []

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
            y_list.append((center[1] - height / 2) / (height / 2))
            area_list.append(area)

        vision_nt.putNumberArray('target_x_yellow', x_list)
        vision_nt.putNumberArray('target_y_yellow', y_list)
        vision_nt.putNumberArray('target_area_yellow', area_list)

        if showColorCluster:
            sf = hsv_img.shape[1] / palette.shape[1]
            palette_bgr = cv2.cvtColor(palette, cv2.COLOR_HSV2BGR)
            palette_block = cv2.resize(palette_bgr, (0, 0), fx=sf, fy=sf, interpolation=cv2.INTER_NEAREST)
            output_img = np.vstack([output_img, palette_block])
            cv2.putText(output_img, "H:" + str(palette[0][0][0]), (80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (00, 0, 0))
            cv2.putText(output_img, "S:" + str(palette[0][0][1]), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))
            cv2.putText(output_img, "V:" + str(palette[0][0][2]), (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))

        if currentInterval > clusterInterval:
            currentInterval = 0

        processing_time = time.time() - start_time
        fps = 1 / processing_time
        cv2.putText(output_img, str(round(fps, 1)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        output_stream.putFrame(output_img)

main()