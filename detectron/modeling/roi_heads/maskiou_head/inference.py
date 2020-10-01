import numpy as np
from jittor import nn,Module
import jittor as jt

from detectron.structures.bounding_box import BoxList


# TODO get the predicted maskiou and mask score.
class MaskIoUPostProcessor(Module):
    """
    Getting the maskiou according to the targeted label, and computing the mask score according to maskiou.
    """

    def __init__(self):
        super(MaskIoUPostProcessor, self).__init__()

    def execute(self, boxes, pred_maskiou, labels):
        num_masks = pred_maskiou.shape[0]
        index = jt.arange(num_masks)
        maskious = pred_maskiou[index, labels]
        # maskious = [maskious]
        # split `maskiou` accroding to `boxes`
        boxes_per_image = [len(box) for box in boxes]
        maskious = maskious.split(boxes_per_image, dim=0)
        results = []
        for maskiou, box in zip(maskious, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox_scores = bbox.get_field("scores")
            mask_scores = bbox_scores * maskiou
            bbox.add_field("mask_scores", mask_scores)
            results.append(bbox)

        return results

def make_roi_maskiou_post_processor(cfg):
    maskiou_post_processor = MaskIoUPostProcessor()
    return maskiou_post_processor
