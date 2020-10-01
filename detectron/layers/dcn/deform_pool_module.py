from jittor import nn,Module,init
import jittor as jt
from .deform_pool_func import deform_roi_pooling


class DeformRoIPooling(Module):

    def __init__(self,
                 spatial_scale,
                 out_size,
                 out_channels,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0):
        super(DeformRoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.out_size = out_size
        self.out_channels = out_channels
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = out_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def execute(self, data, rois, offset):
        if self.no_trans:
            offset = data.new_empty(0)
        return deform_roi_pooling(
            data, rois, offset, self.spatial_scale, self.out_size,
            self.out_channels, self.no_trans, self.group_size, self.part_size,
            self.sample_per_part, self.trans_std)


class DeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self,
                 spatial_scale,
                 out_size,
                 out_channels,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0,
                 deform_fc_channels=1024):
        super(DeformRoIPoolingPack,
              self).__init__(spatial_scale, out_size, out_channels, no_trans,
                             group_size, part_size, sample_per_part, trans_std)

        self.deform_fc_channels = deform_fc_channels

        if not no_trans:
            self.offset_fc = nn.Sequential(
                nn.Linear(self.out_size * self.out_size * self.out_channels,
                          self.deform_fc_channels),
                nn.ReLU(),
                nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
                nn.ReLU(),
                nn.Linear(self.deform_fc_channels,
                          self.out_size * self.out_size * 2))
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()

    def execute(self, data, rois):
        assert data.shape[1] == self.out_channels
        if self.no_trans:
            offset = jt.array([],dtype=str(data.dtype))
            return deform_roi_pooling(
                data, rois, offset, self.spatial_scale, self.out_size,
                self.out_channels, self.no_trans, self.group_size,
                self.part_size, self.sample_per_part, self.trans_std)
        else:
            n = rois.shape[0]
            offset =jt.array([],dtype=str(data.dtype))
            x = deform_roi_pooling(data, rois, offset, self.spatial_scale,
                                   self.out_size, self.out_channels, True,
                                   self.group_size, self.part_size,
                                   self.sample_per_part, self.trans_std)
            offset = self.offset_fc(x.reshape(n, -1))
            offset = offset.reshape(n, 2, self.out_size, self.out_size)
            return deform_roi_pooling(
                data, rois, offset, self.spatial_scale, self.out_size,
                self.out_channels, self.no_trans, self.group_size,
                self.part_size, self.sample_per_part, self.trans_std)


class ModulatedDeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self,
                 spatial_scale,
                 out_size,
                 out_channels,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0,
                 deform_fc_channels=1024):
        super(ModulatedDeformRoIPoolingPack, self).__init__(
            spatial_scale, out_size, out_channels, no_trans, group_size,
            part_size, sample_per_part, trans_std)

        self.deform_fc_channels = deform_fc_channels

        if not no_trans:
            self.offset_fc = nn.Sequential(
                nn.Linear(self.out_size * self.out_size * self.out_channels,
                          self.deform_fc_channels),
                nn.ReLU(),
                nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
                nn.ReLU(),
                nn.Linear(self.deform_fc_channels,
                          self.out_size * self.out_size * 2))
            init.constant_(self.offset_fc[-1].weight)
            init.constant_(self.offset_fc[-1].bias)
            self.mask_fc = nn.Sequential(
                nn.Linear(self.out_size * self.out_size * self.out_channels,
                          self.deform_fc_channels),
                nn.ReLU(),
                nn.Linear(self.deform_fc_channels,
                          self.out_size * self.out_size * 1),
                nn.Sigmoid())
            init.constant_(self.mask_fc[2].weight)
            init.constant_(self.mask_fc[2].bias)

    def execute(self, data, rois):
        assert data.shape[1] == self.out_channels
        if self.no_trans:
            offset = jt.array([],dtype=str(data.dtype))
            return deform_roi_pooling(
                data, rois, offset, self.spatial_scale, self.out_size,
                self.out_channels, self.no_trans, self.group_size,
                self.part_size, self.sample_per_part, self.trans_std)
        else:
            n = rois.shape[0]
            offset = jt.array([],dtype=str(data.dtype))
            x = deform_roi_pooling(data, rois, offset, self.spatial_scale,
                                   self.out_size, self.out_channels, True,
                                   self.group_size, self.part_size,
                                   self.sample_per_part, self.trans_std)
            offset = self.offset_fc(x.reshape(n, -1))
            offset = offset.reshape(n, 2, self.out_size, self.out_size)
            mask = self.mask_fc(x.reshapeview(n, -1))
            mask = mask.reshape(n, 1, self.out_size, self.out_size)
            return deform_roi_pooling(
                data, rois, offset, self.spatial_scale, self.out_size,
                self.out_channels, self.no_trans, self.group_size,
                self.part_size, self.sample_per_part, self.trans_std) * mask
