#coding=utf-8
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import jittor as jt 

jt.flags.use_cuda = 1

pylab.rcParams['figure.figsize'] = 20, 12

from detectron.config import cfg
from predictor import COCODemo


config_file = ['/home/lxl/jittor/maskrcnn-benchmark/configs/caffe2/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/caffe2/e2e_faster_rcnn_R_50_FPN_1x_caffe2.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/caffe2/e2e_faster_rcnn_R_101_FPN_1x_caffe2.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/caffe2/e2e_faster_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_R_50_C4_1x_caffe2.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x_caffe2.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/centermask/centermask_R_101_FPN_ms_2x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/centermask/centermask_R_50_FPN_ms_3x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/centermask/centermask_X_101_FPN_ms_3x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/centermask/centermask_V_99_eSE_FPN_ms_3x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/centermask/centermask_V_57_eSE_FPN_ms_3x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/centermask/centermask_V_39_eSE_FPN_ms_3x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/centermask/centermask_V_19_eSE_FPN_ms_3x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/centermask/centermask_M_v2_FPN_ms_3x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/centermask/centermask_M_v2_FPN_lite_res600_ms_bs16_4x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/maskscoring_rcnn/e2e_ms_rcnn_R_50_FPN_1x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/maskscoring_rcnn/e2e_ms_rcnn_R_101_FPN_1x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/fcos/fcos_R_50_FPN_1x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/fcos/fcos_R_101_FPN_2x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/fcos/fcos_X_101_32x8d_FPN_2x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/fcos/fcos_X_101_64x4d_FPN_2x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/fcos/fcos_bn_bs16_MNV2_FPN_1x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/fcos/fcos_imprv_R_50_FPN_1x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/fcos/fcos_imprv_R_101_FPN_2x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/fcos/fcos_syncbn_bs32_c128_MNV2_FPN_1x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/fcos/fcos_syncbn_bs32_c128_ms_MNV2_FPN_1x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/embedmask/embed_mask_R50_1x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/embedmask/embed_mask_R101_1x.yaml',
               '/home/lxl/jittor/maskrcnn-benchmark/configs/embedmask/embed_mask_R101_ms_3x.yaml',
              ][0]

# update the config options with the config file
cfg.merge_from_file(config_file)
#cfg.MODEL.WEIGHT = "./centermask_weights/centermask-R-101-ms-2x.pth"

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.3,
)

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

image = load("http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")

# compute predictions
predictions = coco_demo.run_on_opencv_image(image)
cv2.imwrite('direct.jpg',predictions)
