import cv2
from realsense_depth import *

#Initializing Camera and getting intrinsic parameters
camera = DepthCamera()

#To calculate average depth for smoothness
depth_array = []

run = True

while run:
    try:
        #Getting photo footage
        ret, depth_frame, color_frame = camera.get_frame()

        cam_number = 1
        for i in range(len(camera.pipeline)):

            if depth_array == []:
                depth_array = [[] for _ in range(len(camera.pipeline))]

            #Resizing the actual depth frame due to "decimation" post-processing and filtering techniqe in camera.get_frame()
            current_depth_frame = cv2.resize(depth_frame[i], (640,480), cv2.INTER_CUBIC)

            current_depth_frame = np.clip(current_depth_frame,0,8000)
            current_depth_frame = current_depth_frame*1.8

            depth_array[i].append(current_depth_frame)

            #Preserving last 10 frames
            if len(depth_array[i]) > 10:
                depth_array[i].pop(0)

            #Getting average depth across previoud 10 frames
            avg_depth = sum(depth_array[i])/10

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(avg_depth, alpha=0.03), cv2.COLORMAP_JET)

            cv2.imshow("Depth Video Camera {}".format(cam_number), depth_colormap)
            cv2.imshow("Color Video Camera {}".format(cam_number), color_frame[i])
            cam_number += 1
        cv2.waitKey(1)

    #Press Ctrl+C in terminal to stop the program
    except KeyboardInterrupt:
        print("exit")
        run = False
        pass