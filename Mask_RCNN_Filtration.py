from shapely.geometry import Polygon
from Information_Processing_Utilities import *
from Model_Processing_Utilities import *
from Filtration_Utilities import *
from mmdet.apis import inference_detector
import mmcv

#Names of the instance segmentation models being used
mushroom_architecture_selected = "mushroom_mask_rcnn"
substrate_architecture_selected = "substrate_mask_rcnn"

# Set the paths for differnt folders
working_folder = "Results/"
configs_folder = "configs/"
predicted_images = working_folder + 'predicted_images/'

#Path to images
test_set_path = "Images/Timelapse_2/test/"

# Creating the output folders
os.makedirs(working_folder,exist_ok=True)
os.makedirs(predicted_images,exist_ok=True)
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

        test_img = 'Test_Img ({}).JPG'.format(img_num+1)

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

        print('Image {}'.format(img_num+1))
