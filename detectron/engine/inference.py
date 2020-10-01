# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

from tqdm import tqdm
import jittor as jt
import torch

from detectron.data.datasets.evaluation import evaluate
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug
import datetime


def compute_on_dataset(model, data_loader, bbox_aug, timer=None):
    model.eval()
    results_dict = {}
    #jt.profiler.start(0, 0)
    for i, batch in enumerate(tqdm(data_loader)):
        #if i<125:continue
        #jt.sync_all()
        #print(1,time.asctime())
        jt.display_memory_info()
        images, targets, image_ids = batch
        images.tensors = images.tensors.float32()
        print(images.tensors.shape)
        #print(2,time.asctime())

        with jt.no_grad():
            if timer:
                timer.tic()
            if bbox_aug:
                output = im_detect_bbox_aug(model, images)
            else:
                output = model(images)
            if timer:
                timer.toc()
        #jt.sync_all()
        #print(7,time.asctime())
        jt.fetch(image_ids, output, lambda image_ids, output: \
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
        )
        jt.sync_all(True)


    #jt.profiler.stop()
    #jt.profiler.report()

    return results_dict


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        bbox_aug=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    logger = logging.getLogger("detectron.inference")
    dataset = data_loader
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, bbox_aug, inference_timer)
    # wait for all processes to complete before measuring the time
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * 1 / len(dataset), 1
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time / len(dataset),
            1,
        )
    )

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
