# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

from tqdm import tqdm
import jittor as jt

from detectron.data.datasets.evaluation import evaluate
from detectron.structures.image_list import to_image_list,ImageList

import detectron.utils.timer as test_timer
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug
import datetime

def detach_output(output):
    for o in output:
        o.bbox = o.bbox.detach()
        o.bbox.sync()
        for k in o.extra_fields:
            o.extra_fields[k] = o.extra_fields[k].detach()
            o.extra_fields[k].sync()
    return output

from jittor.utils.nvtx import nvtx_scope

# jt.cudnn.set_algorithm_cache_size(0)

def compute_on_dataset(model, data_loader, bbox_aug, timer=None):
    model.eval()
    results_dict = {}
    data_loader.is_train=False
    data_loader.num_workers = 4
    start_time = 0
    import cProfile as profiler
    for i, batch in enumerate(tqdm(data_loader)):
        if i==20:
            # For fair comparison,remove jittor compiling time 
            start_time = time.time()
            # jt.profiler.start()

        with nvtx_scope("preprocess"):
            images, image_sizes, image_ids = batch
            # images = ImageList(jt.array(images),image_sizes)
        with nvtx_scope("model"):
            with jt.no_grad():
                if timer:
                    timer.tic()
                if bbox_aug:
                    output = im_detect_bbox_aug(model, images)
                else:
                    output = model(images)
                if timer:
                    timer.toc()
        with nvtx_scope("detach"):
            output = detach_output(output)
            results_dict.update(
                    {img_id: result for img_id, result in zip(image_ids, output)}
                )
    
    end_time = time.time()
    print('fps',(5000-20*data_loader.batch_size)/(end_time-start_time))

    return results_dict

def convert_numpy(data):
    if hasattr(data,'cpu'):
        data = data.cpu()
    if hasattr(data,'numpy'):
        return data.numpy()
    return data

def save_predictions(predictions):
    for img_id,o in predictions.items():
        o.bbox = convert_numpy(o.bbox)
        for k in o.extra_fields:
            o.extra_fields[k] = convert_numpy(o.extra_fields[k])
    import pickle
    pickle.dump(predictions,open('/home/lxl/tmp/predictions_jt.pkl','wb'))
        

def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        bbox_aug=False,
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        cfg = None
):
    logger = logging.getLogger("detectron.inference")

    dataset = data_loader
    logger.warn("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, bbox_aug, inference_timer)
    # wait for all processes to complete before measuring the time
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.warn(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * 1 / len(dataset), 1
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.warn(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time / len(dataset),
            1,
        )
    )

    if output_folder:
        jt.save(predictions, os.path.join(output_folder, "predictions.pkl"))
    # save_predictions(predictions)

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    # return None
    # return evaluate(dataset=dataset,
    #                 predictions=predictions,
    #                 output_folder=output_folder,
    #                 **extra_args)
