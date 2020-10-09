#coding=utf-8
import requests
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import jittor as jt 
from detectron.config import cfg
from predictor import COCODemo

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

jt.flags.use_cuda = 1

config_file = '/home/lxl/jittor/detectron.jittor/configs/maskrcnn_benchmark/e2e_mask_rcnn_R_50_FPN_1x.yaml'
# update the config options with the config file
cfg.merge_from_file(config_file)
#cfg.MODEL.WEIGHT = "weight/maskrcnn_r50.pth"

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.3,
)

image = load("http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")

# compute predictions
predictions = coco_demo.run_on_opencv_image(image)
cv2.imwrite('predicton1.jpg',predictions)
