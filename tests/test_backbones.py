# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest
import copy
import jittor as jt
# import modules to to register backbones
from detectron.modeling.backbone import build_backbone # NoQA
from detectron.modeling import registry
from detectron.config import cfg as g_cfg
from utils import load_config


# overwrite configs if specified, otherwise default config is used
BACKBONE_CFGS = {
    "R-50-FPN": "e2e_faster_rcnn_R_50_FPN_1x.yaml",
    "R-101-FPN": "e2e_faster_rcnn_R_101_FPN_1x.yaml",
    "R-152-FPN": "e2e_faster_rcnn_R_101_FPN_1x.yaml",
    "R-50-FPN-RETINANET": "retinanet/retinanet_R-50-FPN_1x.yaml",
    "R-101-FPN-RETINANET": "retinanet/retinanet_R-101-FPN_1x.yaml",
}


class TestBackbones(unittest.TestCase):
    def test_build_backbones(self):
        ''' Make sure backbones run '''

        self.assertGreater(len(registry.BACKBONES), 0)

        for name, backbone_builder in registry.BACKBONES.items():
            print('Testing {}...'.format(name))
            if name in BACKBONE_CFGS:
                cfg = load_config(BACKBONE_CFGS[name])
            else:
                # Use default config if config file is not specified
                cfg = copy.deepcopy(g_cfg)
            backbone = backbone_builder(cfg)

            # make sures the backbone has `out_channels`
            self.assertIsNotNone(
                getattr(backbone, 'out_channels', None),
                'Need to provide out_channels for backbone {}'.format(name)
            )

            N, C_in, H, W = 2, 3, 224, 256
            input = jt.rand([N, C_in, H, W]).float32()
            out = backbone(input)
            for cur_out in out:
                self.assertEqual(
                    cur_out.shape[:2],
                    ([N, backbone.out_channels])
                )


if __name__ == "__main__":
    unittest.main()
