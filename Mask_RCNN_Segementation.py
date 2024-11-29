from shapely.geometry import Polygon
from Information_Processing_Utilities import *
from Model_Processing_Utilities import *
from mmdet.apis import inference_detector
import mmcv
import os

#Names of the instance segmentation models being used
mushroom_architecture_selected = "mushroom_mask_rcnn"
substrate_architecture_selected = "substrate_mask_rcnn"

# Set the paths for differnt folders
working_folder = "Results/"
configs_folder = "configs/"
predicted_images = working_folder + 'predicted_images/'

#Path to images
test_set_path = "Images/Timelapse_1/test/"

# Creating the output folders
os.makedirs(working_folder,exist_ok=True)
os.makedirs(predicted_images,exist_ok=True)
os.makedirs(working_folder + "/Substrate/",exist_ok=True)

#Checking for available cuda/cpu
use_device = check_cuda()

#Loading prediction models
mushroom_model,substrate_model,visualizer = load_models(configs_folder,mushroom_architecture_selected,substrate_architecture_selected,use_device)

#Confidence thresholds
confidence_score_threshold = 0.5

for img_num in range(len(os.listdir(test_set_path))):
    #To control which images are being processed
    if img_num > -1 and img_num < 300:

        test_img = 'Test_Img ({}).JPG'.format(img_num+1)

        # load the image
        img = mmcv.imread(test_set_path + test_img)

        # Mushroom segmentation inference
        image_result = inference_detector(mushroom_model, img)

        #Color correction of the images
        img = mmcv.image.bgr2rgb(img)

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

        print('Image {}'.format(img_num+1))
        
