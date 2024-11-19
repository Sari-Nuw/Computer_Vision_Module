import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from ultralytics import YOLO
import os

def bbox_2_yolo(x1,y1,x2,y2,width,height):

    #Calculating parameters and normalzing
    bbox_width = (x2 - x1)/width
    bbox_height = (y2 - y1)/height

    x_center = x1/width + bbox_width/2
    y_center = y1/height + bbox_height/2

    return x_center,y_center,bbox_width,bbox_height

#Path to images for prediction
image_path = r"C:/Users/nuway/OneDrive/Desktop/GOOD_Project/GOOD_Dataset/Images"

#Path to result images
results_path = r"C:/Users/nuway/OneDrive/Desktop/GOOD_Project/Prediction Results/"

#Path to object detection model
model_path = r'runs\detect\epoch75n\weights\best.pt'

# create the result folders
os.makedirs(image_path,exist_ok=True)
os.makedirs(results_path,exist_ok=True)

# Load a model
model = YOLO(model_path)

#Setting threshold for object detection 
threshold = 0.5

#Reading image
for num in range(7):

    color_image = mpimg.imread(image_path + 'img ({})'.format(num))
    #Gets the color bands from the images and also the binary of the image (whether or not data is available)
    copy_image = np.copy(color_image)

    #Getting results of object detection
    results = model(color_image)[0]

    #Pixel coordinates of objects from predictions
    prediction_boxes = []

    #Saving the position of the object boxes from predictions
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            #Storing all the predicted boxes
            points = [int(x1),int(y1),int(x2),int(y2)]
            prediction_boxes.append(points)
            #To see bounding box around detected object
            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)

    #Show the image with bounding boxes
    plt.imshow(color_image)
    plt.show()


