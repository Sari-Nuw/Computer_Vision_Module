import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import os

#Path to images for prediction
image_path = r"Images/Timelapse_1/test/"

#Path to result images
results_path = r"Results/YOLO_Images/"

#Path to object detection model
model_path = r"runs\detect\train\weights\best.pt"

# create the result folders
os.makedirs(image_path,exist_ok=True)
os.makedirs(results_path,exist_ok=True)

# Load a model
model = YOLO(model_path)

#Setting threshold for object detection 
threshold = 0.01

#Reading image
for num in range(20):

    color_image = cv2.imread(image_path + 'Test_Img ({}).JPG'.format(num+1))
    color_image = cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)

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

    #Saving results
    cv2.imwrite(results_path + "Test_Img_YOLO_Detections ({}).JPG".format(num+1),cv2.cvtColor(color_image,cv2.COLOR_RGB2BGR))

def bbox_2_yolo(x1,y1,x2,y2,width,height):

    #Calculating parameters and normalzing
    bbox_width = (x2 - x1)/width
    bbox_height = (y2 - y1)/height

    x_center = x1/width + bbox_width/2
    y_center = y1/height + bbox_height/2

    return x_center,y_center,bbox_width,bbox_height


