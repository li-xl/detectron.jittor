# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest
import glob
import os
import copy
import jittor as jt
from detectron.modeling.detector import build_detection_model
from detectron.structures.image_list import to_image_list
import utils


CONFIG_FILES = [
    # bbox
    "e2e_faster_rcnn_R_50_C4_1x.yaml",
    "e2e_faster_rcnn_R_50_FPN_1x.yaml",
    "e2e_faster_rcnn_fbnet.yaml",

    # mask
    "e2e_mask_rcnn_R_50_C4_1x.yaml",
    "e2e_mask_rcnn_R_50_FPN_1x.yaml",
    "e2e_mask_rcnn_fbnet.yaml",

    # keypoints
    # TODO: fail to run for random model due to empty head input
    # "e2e_keypoint_rcnn_R_50_FPN_1x.yaml",

    # gn
    "gn_baselines/e2e_faster_rcnn_R_50_FPN_1x_gn.yaml",
    # TODO: fail to run for random model due to empty head input
    # "gn_baselines/e2e_mask_rcnn_R_50_FPN_Xconv1fc_1x_gn.yaml",
	
    # retinanet
    "retinanet/retinanet_R-50-FPN_1x.yaml",

    # rpn only
    "rpn_R_50_C4_1x.yaml",
    "rpn_R_50_FPN_1x.yaml",
]

EXCLUDED_FOLDERS = [
    "caffe2",
    "quick_schedules",
    "pascal_voc",
    "cityscapes",
]


TEST_CUDA = True


def get_config_files(file_list, exclude_folders):
    cfg_root_path = utils.get_config_root_path()
    if file_list is not None:
        files = [os.path.join(cfg_root_path, x) for x in file_list]
    else:
        files = glob.glob(
            os.path.join(cfg_root_path, "./**/*.yaml"), recursive=True)

    def _contains(path, exclude_dirs):
        return any(x in path for x in exclude_dirs)

    if exclude_folders is not None:
        files = [x for x in files if not _contains(x, exclude_folders) and 'hrnet' not in x]

    return files


def create_model(cfg):
    cfg = copy.deepcopy(cfg)
    cfg.freeze()
    model = build_detection_model(cfg)
    return model


def create_random_input(cfg):
    ret = []
    for x in cfg.INPUT.MIN_SIZE_TRAIN:
        ret.append(jt.random((3, x, int(x * 1.2))))
    ret = to_image_list(ret, cfg.DATALOADER.SIZE_DIVISIBILITY)
    return ret


def _test_build_detectors(self):
    ''' Make sure models build '''

    cfg_files = get_config_files(None, EXCLUDED_FOLDERS)
    self.assertGreater(len(cfg_files), 0)

    for cfg_file in cfg_files:
        with self.subTest(cfg_file=cfg_file):
            print('Testing {}...'.format(cfg_file))
            cfg = utils.load_config_from_file(cfg_file)
            create_model(cfg)


def _test_run_selected_detectors(self, cfg_files):
    ''' Make sure models build and run '''
    self.assertGreater(len(cfg_files), 0)

    for cfg_file in cfg_files:
        with self.subTest(cfg_file=cfg_file):
            print('Testing {}...'.format(cfg_file))
            cfg = utils.load_config_from_file(cfg_file)
            cfg.MODEL.RPN.POST_NMS_TOP_N_TEST = 10
            cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 10
            model = create_model(cfg)
            inputs = create_random_input(cfg)
            model.eval()
            output = model(inputs)
            self.assertEqual(len(output), len(inputs.image_sizes))


class TestDetectors(unittest.TestCase):
    def test_build_detectors(self):
        ''' Make sure models build '''
        jt.flags.use_cuda=0
        _test_build_detectors(self)

    @unittest.skipIf(not TEST_CUDA, "no CUDA detected")
    def test_build_detectors_cuda(self):
        ''' Make sure models build on gpu'''
        jt.flags.use_cuda=1
        _test_build_detectors(self)

    def test_run_selected_detectors(self):
        ''' Make sure models build and run '''
        jt.flags.use_cuda=0
        # run on selected models
        cfg_files = get_config_files(CONFIG_FILES, None)
        # cfg_files = get_config_files(None, EXCLUDED_FOLDERS)
        _test_run_selected_detectors(self, cfg_files)

    @unittest.skipIf(not TEST_CUDA, "no CUDA detected")
    def test_run_selected_detectors_cuda(self):
        ''' Make sure models build and run on cuda '''
        jt.flags.use_cuda=1
        # run on selected models
        cfg_files = get_config_files(CONFIG_FILES, None)
        # cfg_files = get_config_files(None, EXCLUDED_FOLDERS)
        _test_run_selected_detectors(self, cfg_files)


if __name__ == "__main__":
    unittest.main()
