import cv2
from realsense_depth import *
import math
import time
from datetime import datetime
import numpy as np
from DepthMap import * 
from PointCloud import *
from MonocularDepthEstimation import *
import os

#Initializing Camera and getting intrinsic parameters
camera = DepthCamera()
camera_intrinsics = [camera.intrinsics_width, camera.intrinsics_height, camera.intrinsics_fx, camera.intrinsics_fy, camera.intrinsics_ppx, camera.intrinsics_ppy]

#Skipping initial frames
time.sleep(3)

#Control how often an image is taken in minutes (interval = number of minutes)
interval = 30
wait = math.floor(interval*60)

#Expected minimum and maximum distances for the camera/image (in mm)
min_dist = 0
max_dist = 2000

#Online upload
online_upload = False

#Generate Point Clouds
pcd_gen = True

#Path under which images are saved
path = r'.\Image_Capture\\'
#Checking that path folder exists to save images locally. Otherwise create the pathway.
if not os.path.exists(path):
    os.mkdir(path)
if not os.path.exists(path+'Picture\\'):
    os.mkdir(path+'Picture\\')
if not os.path.exists(path+'Stereo Depth Map\\'):
    os.mkdir(path+'Stereo Depth Map\\')
if not os.path.exists(path+'Estimated Depth Map\\'):
    os.mkdir(path+'Estimated Depth Map\\')
if not os.path.exists(path+'Estimated PCD\\'):
    os.mkdir(path+'Estimated PCD\\')
if not os.path.exists(path+'Normalized Stereo PCD\\'):
    os.mkdir(path+'Normalized Stereo PCD\\')

while True:
    try:
        #Start measuring time needed for image processing and upload
        start_time = time.time() 

        #Getting photo footage
        ret, depth_frames, color_frames = camera.get_frame()

        cam_number = 1
        for i in range(len(camera.pipeline)):

            #Getting date and time
            now = datetime.now()
            now_str = now.strftime("Date %Y-%m-%d Time %H;%M;%S Camera {}".format(cam_number))

            #Getting the color and depth frames from each individual camera
            depth_frame = depth_frames[i]
            color_frame = color_frames[i]
            #Getting the parameters of each individual camera
            camera_intrinsic = [parameter[i] for parameter in camera_intrinsics]

            #Resizing the actual depth frame due to "decimation" post-processing and filtering techniqe in camera.get_frame()
            depth_frame = cv2.resize(depth_frame, (640,480), cv2.INTER_CUBIC)

            #Estimating the depth frame from the color_frame image
            monocular_depth_frame = MonocularDepthEstimation(color_frame, min_dist, max_dist)

            #Saving image
            written = cv2.imwrite(path+"Picture\Picture {}.png".format(now_str), color_frame)

            #Generating and saving the stereo depth map
            depth_img = DepthMap(depth_frame, min_dist, max_dist)
            depth_img.save(path+"Stereo Depth Map\Stereo Depth Map {}.png".format(now_str))
            
            #Generating and saving the estimated depth map
            estimated_depth_img = DepthMap(monocular_depth_frame, min_dist, max_dist)
            estimated_depth_img.save(path+"Estimated Depth Map\Estimated Depth Map {}.png".format(now_str))

            #Generating point clouds from color and estimated depth images
            if pcd_gen:
                estimated_pcd = PointCloud(color_frame,monocular_depth_frame.astype('uint16'), camera_intrinsic)
                o3d.io.write_point_cloud(path+"Estimated PCD\Estimated PCD {}.pcd".format(now_str), estimated_pcd)

                #Making a copy of the camera depth frame and nomalizing it between 0-2000 for point cloud plotting
                depth_frame_copy = np.copy(depth_frame)
                depth_frame_copy = (max_dist - min_dist)*((depth_frame_copy - np.min(depth_frame_copy))/(np.max(depth_frame_copy) - np.min(depth_frame_copy)))
                depth_frame_copy = depth_frame_copy.astype('uint16')
                
                #Generating point cloud from color and RealSense camera normalized depth images
                normalized_stereo_pcd = PointCloud(color_frame,depth_frame_copy, camera_intrinsic)
                o3d.io.write_point_cloud(path+"Normalized Stereo PCD\\Normalized Stereo PCD {}.pcd".format(now_str), normalized_stereo_pcd)

            print("uploaded")

            cam_number +=1

        #Complete measuring time needed for image processing and upload
        end_time = time.time()
        elapsed_time = end_time - start_time

        #Change wait time to account for image processing and upload
        waittime = math.floor(wait - elapsed_time)
        #For processing time greater than wait time
        if waittime < 0:
            waittime = 0

        #Waiting to take another image
        time.sleep(waittime)

    #Press Ctrl+C in terminal to stop the program
    except KeyboardInterrupt:
        print("exit")
        break