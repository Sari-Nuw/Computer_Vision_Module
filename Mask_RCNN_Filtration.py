from shapely.geometry import Polygon
from Information_Processing_Utilities import *
from Model_Processing_Utilities import *
from Filtration_Utilities import *
from mmdet.apis import inference_detector
import mmcv

#Names of the instance segmentation models being used
mushroom_architecture_selected = "mushroom_custom_config_mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco"
substrate_architecture_selected = "substrate_custom_config_mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco"

# Set the paths for differnt folders
working_folder = "./results/" + mushroom_architecture_selected + "/"
configs_folder = "./configs/"
predicted_images = working_folder + 'predicted_images/'

#Path to images
test_set_path = r"C:/Users/nuway/OneDrive/Desktop/Realsense Project/Python_Marigold/Timelapse/Timelapse1//"

# Creating the output folders
os.makedirs(working_folder,exist_ok=True)
os.makedirs(predicted_images,exist_ok=True)
os.makedirs(working_folder + "/Annotated/",exist_ok=True)
os.makedirs(working_folder + "/Unsorted/",exist_ok=True)
os.makedirs(working_folder + "/Substrate/",exist_ok=True)

#Checking for available cuda/cpu
use_device = check_cuda()

#Loading prediction models
mushroom_model,substrate_model,visualizer = load_models(configs_folder,mushroom_architecture_selected,substrate_architecture_selected,use_device)

#Tracking images
images = []

#Tracking clusters and cluster information 
polygons = []
polygons_info = []

#Baseline for sorting
baseline = []

#Saving pixel length of the substrate in images
detected_length_pixels = []
averaged_length_pixels = []

#Confidence thresholds
confidence_score_threshold = 0.5
overlapping_iou_threshold = 0.2

for img_num in range(len(os.listdir(test_set_path))):
    #To control which images are being processed
    if img_num > -1 and img_num < 300:

        test_img = 'img ({}).JPG'.format(img_num+1)

        # load the image
        img = mmcv.imread(test_set_path + test_img)
        substrate_img = img.copy()

        #Substrate segmentation inference
        substrate_result = inference_detector(substrate_model, img).pred_instances
                
        # calculate substrate length data
        detected_length_pixels.append(substrate_result[0]["bboxes"].cpu().numpy()[0][2] - substrate_result[0]["bboxes"].cpu().numpy()[0][0])

        # calculate the substrate length average
        averaged_length_pixels.append(sum(detected_length_pixels)/len(detected_length_pixels))

        # Mushroom segmentation inference
        image_result = inference_detector(mushroom_model, img)

        #Color correction of the images
        img = mmcv.image.bgr2rgb(img)

        # Chcecking substrate results and saving substrate image
        process_substrate_results(img,substrate_result,working_folder,img_num)

        #saving image for processing and image file names 
        images.append(img)

        # show the results before filtering
        visualizer.add_datasample(
            'result',
            img,
            data_sample=image_result,
            draw_gt = None,
            wait_time=0,
            out_file=predicted_images + "before_filtration_prediction_" + test_img,
            pred_score_thr=confidence_score_threshold
        )

        # Result filters
        image_result = delete_low_confidence_predictions(image_result,confidence_score_threshold)
        image_result = delete_overlapping_with_lower_confidence(image_result,overlapping_iou_threshold)
        image_result = simple_delete_post_background_clusters(image_result,substrate_result)

        #Processing of reuslts for use in different data structures
        results, results_info = process_results(image_result,averaged_length_pixels,substrate_real_size = 50)     

        #Saving the hull results for all the clusters in the image
        polygons.append(results)
        polygons_info.append(results_info)

        # show the results after filtering
        visualizer.add_datasample(
            'result',
            img,
            data_sample=image_result,
            draw_gt = None,
            wait_time=0,
            out_file=predicted_images + "after_filtration_prediction_" + test_img,
            pred_score_thr=confidence_score_threshold
        )

        #Copying the current image for processing
        image_copy = (img, cv2.COLOR_RGB2BGR)[0]
        full_image = np.copy(img)
        
        #Saving the images with predicted polygons
        #Polygons from current image
        j = 0
        for poly in polygons[-1]:
            #Draw lines
            if len(poly) > 1:
                poly.reshape(-1,1,2)
                #Getting the centre point of the polygons
                centre = Polygon(poly).centroid

                #Saving the image with outlined clusters
                cv2.polylines(full_image, np.int32([poly]), True, (255, 0, 0), 10)
                #cv2.putText(full_image, '{} {}'.format(j,poly[0]), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 4, (0,255,0), 6, cv2.LINE_AA)
                cv2.putText(full_image, '{}'.format(j), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 4, (0,255,0), 6, cv2.LINE_AA)

            j += 1    

        #Saving image in various forms
        save_image(working_folder,full_image,img_num)

        print('Image {}'.format(img_num+1))
