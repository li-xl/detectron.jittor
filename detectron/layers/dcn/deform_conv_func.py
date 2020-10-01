import jittor as jt
from jittor import Function
from jittor.misc import _pair

class DeformConvFunction(Function):

    def execute(
        self,
        input,
        offset,
        weight,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        im2col_step=64
    ):
        if input is not None and input.ndim != 4:
            raise ValueError(
                "Expected 4D tensor as input, got {}D tensor instead.".format(
                    input.ndim))
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step

        self.saved_tensors = (input, offset, weight)

        output = jt.zeros(
            self._output_size(input, weight, self.padding,
                                            self.dilation, self.stride),dtype=input.dtype)

        self.bufs_ = [jt.zeros((0),dtype=input.dtype), jt.zeros((0),dtype=input.dtype)]  # columns, ones

        cur_im2col_step = min(self.im2col_step, input.shape[0])
        assert (input.shape[0] % cur_im2col_step) == 0, 'im2col step must divide batchsize'
        '''_C.deform_conv_forward(
                input,
                weight,
                offset,
                output,
                self.bufs_[0],
                self.bufs_[1],
                weight.size(3),
                weight.size(2),
                self.stride[1],
                self.stride[0],
                self.padding[1],
                self.padding[0],
                self.dilation[1],
                self.dilation[0],
                self.groups,
                self.deformable_groups,
                cur_im2col_step
            )
        '''
        return output

    def grad(self, grad_output):
        input, offset, weight = self.saved_tensors

        grad_input = grad_offset = grad_weight = None

        cur_im2col_step = min(self.im2col_step, input.shape[0])
        assert (input.shape[0] % cur_im2col_step) == 0, 'im2col step must divide batchsize'
        '''
            if self.needs_input_grad[0] or self.needs_input_grad[1]:
                grad_input = jt.zeros_like(input)
                grad_offset = jt.zeros_like(offset)
                _C.deform_conv_backward_input(
                    input,
                    offset,
                    grad_output,
                    grad_input,
                    grad_offset,
                    weight,
                    self.bufs_[0],
                    weight.size(3),
                    weight.size(2),
                    self.stride[1],
                    self.stride[0],
                    self.padding[1],
                    self.padding[0],
                    self.dilation[1],
                    self.dilation[0],
                    self.groups,
                    self.deformable_groups,
                    cur_im2col_step
                )

            if self.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                _C.deform_conv_backward_parameters(
                    input,
                    offset,
                    grad_output,
                    grad_weight,
                    self.bufs_[0],
                    self.bufs_[1],
                    weight.size(3),
                    weight.size(2),
                    self.stride[1],
                    self.stride[0],
                    self.padding[1],
                    self.padding[0],
                    self.dilation[1],
                    self.dilation[0],
                    self.groups,
                    self.deformable_groups,
                    1,
                    cur_im2col_step
                )
               '''
        return (grad_input, grad_offset, grad_weight, None, None, None, None, None)

    def _output_size(self,input, weight, padding, dilation, stride):
        channels = weight.shape[0]
        output_size = (input.shape[0], channels)
        for d in range(input.ndim - 2):
            in_size = input.shape[d + 2]
            pad = padding[d]
            kernel = dilation[d] * (weight.shape[d + 2] - 1) + 1
            stride_ = stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        return output_size


class ModulatedDeformConvFunction(Function):

    def execute(
        self,
        input,
        offset,
        mask,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1
    ):
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias is not None
        if not self.with_bias:
            bias = jt.zeros((1),dtype=input.dtype)  # fake tensor
        if not weight.is_stop_grad() or not mask.is_stop_grad() or not offset.is_stop_grad() or not input.is_stop_grad():
            self.saved_tensors = (input, offset, mask, weight, bias)
        output = input.new_empty(
            self._infer_shape(input, weight))
        self._bufs = [input.new_empty(0), input.new_empty(0)]
        '''
        _C.modulated_deform_conv_forward(
            input,
            weight,
            bias,
            self._bufs[0],
            offset,
            mask,
            output,
            self._bufs[1],
            weight.shape[2],
            weight.shape[3],
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation,
            self.groups,
            self.deformable_groups,
            self.with_bias
        )'''
        return output

    def grad(self, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = self.saved_tensors
        grad_input = jt.zeros_like(input)
        grad_offset = jt.zeros_like(offset)
        grad_mask = jt.zeros_like(mask)
        grad_weight = jt.zeros_like(weight)
        grad_bias = jt.zeros_like(bias)
        '''
        _C.modulated_deform_conv_backward(
            input,
            weight,
            bias,
            self._bufs[0],
            offset,
            mask,
            self._bufs[1],
            grad_input,
            grad_weight,
            grad_bias,
            grad_offset,
            grad_mask,
            grad_output,
            weight.shape[2],
            weight.shape[3],
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation,
            self.groups,
            self.deformable_groups,
            self.with_bias
        )
        '''
        if not self.with_bias:
            grad_bias = None

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None)

    def _infer_shape(self, input, weight):
        n = input.shape[0]
        channels_out = weight.shape[0]
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * self.padding -
                      (self.dilation * (kernel_h - 1) + 1)) // self.stride + 1
        width_out = (width + 2 * self.padding -
                     (self.dilation * (kernel_w - 1) + 1)) // self.stride + 1
        return n, channels_out, height_out, width_out


deform_conv = DeformConvFunction()
modulated_deform_conv = ModulatedDeformConvFunction()
