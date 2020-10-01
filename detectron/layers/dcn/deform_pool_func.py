import jittor as jt 
from jittor import Function


class DeformRoIPoolingFunction(Function):

    def execute(
        self,
        data,
        rois,
        offset,
        spatial_scale,
        out_size,
        out_channels,
        no_trans,
        group_size=1,
        part_size=None,
        sample_per_part=4,
        trans_std=.0
    ):
        self.spatial_scale = spatial_scale
        self.out_size = out_size
        self.out_channels = out_channels
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = out_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

        assert 0.0 <= self.trans_std <= 1.0
        if not data.is_cuda:
            raise NotImplementedError

        n = rois.shape[0]
        output = data.new_empty(n, out_channels, out_size, out_size)
        output_count = data.new_empty(n, out_channels, out_size, out_size)
        '''
        _C.deform_psroi_pooling_forward(
            data,
            rois,
            offset,
            output,
            output_count,
            self.no_trans,
            self.spatial_scale,
            self.out_channels,
            self.group_size,
            self.out_size,
            self.part_size,
            self.sample_per_part,
            self.trans_std
        )
        '''

        if data.requires_grad or rois.requires_grad or offset.requires_grad:
            self.saved_tensors = (data, rois, offset)
        self.output_count = output_count

        return output

    def grad(self, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError

        data, rois, offset = self.saved_tensors
        output_count = self.output_count
        grad_input = jt.zeros_like(data)
        grad_rois = None
        grad_offset = jt.zeros_like(offset)
        '''
        _C.deform_psroi_pooling_backward(
            grad_output,
            data,
            rois,
            offset,
            output_count,
            grad_input,
            grad_offset,
            self.no_trans,
            self.spatial_scale,
            self.out_channels,
            self.group_size,
            self.out_size,
            self.part_size,
            self.sample_per_part,
            self.trans_std
        )
        '''
        return (grad_input, grad_rois, grad_offset, None, None, None, None, None, None, None, None)


deform_roi_pooling = DeformRoIPoolingFunction
