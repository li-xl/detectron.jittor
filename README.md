# Detectron.jittor
This is a repo converted from maskrcnn-benchmark and added more SOFA models based on jittor

We reimplemented maskrcnn-benchmark and added more SOFA models into it. maskrcnn-benchmark is [here](https://github.com/facebookresearch/maskrcnn-benchmark)

Why not reimplement detectron2?
Because it is based on a lot of additional libs and we want a pure repo.

### Usage

**edit**

detectron/config/paths_catalog.py

```python
class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
```
**install**

python setup.py install

**Edit config**
```
MODEL:
  META_ARCHITECTURE: "GeneralizedRCNN"
  WEIGHT: "https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_C4_1x.pth"
  RPN:
    PRE_NMS_TOP_N_TEST: 6000
    POST_NMS_TOP_N_TEST: 1000
  ROI_MASK_HEAD:
    PREDICTOR: "MaskRCNNC4Predictor"
    SHARE_BOX_FEATURE_EXTRACTOR: True
  MASK_ON: True
DATASETS:
  TRAIN: ("coco_2014_train", "coco_2014_valminusminival")
  TEST: ("coco_2014_minival",)
SOLVER:
  BASE_LR: 0.01
  WEIGHT_DECAY: 0.0001
  STEPS: (120000, 160000)
  MAX_ITER: 180000
  IMS_PER_BATCH: 8
```

**Examples**
```python
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

# turn on cuda
jt.flags.use_cuda = 1

# set config
config_file = '../configs/maskrcnn_benchmark/e2e_mask_rcnn_R_50_FPN_1x.yaml'
# update the config options with the config file
cfg.merge_from_file(config_file)
#cfg.MODEL.WEIGHT = "weight/maskrcnn_r50.pth"

#set predictor
coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
)

#load image
pil_image = Image.open('test.jpg').convert("RGB")
image = np.array(pil_image)[:, :, [2, 1, 0]]

# compute predictions
predictions = coco_demo.run_on_opencv_image(image)

# save result
cv2.imwrite('predicton.jpg',predictions)

 ```
 
**Train**:
```shell
python tools/train_net.py --config-file /path/your/configfile
```
**Test**:
```shell
python tools/test_net.py --config-file /path/your/configfile
```

### Reference
1. https://github.com/facebookresearch/maskrcnn-benchmark

