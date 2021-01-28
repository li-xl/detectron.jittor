# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import division

import jittor as jt
import numpy as np

class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)


def to_image_list(tensors, size_divisible=0):
    """
    tensors can be an ImageList, a jt.Var or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors,np.ndarray):
        tensors = jt.array(tensors)
    if isinstance(tensors, jt.Var) and size_divisible > 0:
        tensors = [tensors]

    if isinstance(tensors, ImageList):
        if not isinstance(tensors.tensors,jt.Var):
            tensors.tensors = jt.array(tensors.tensors)
        return tensors
    elif isinstance(tensors, jt.Var):
        # single tensor shape can be inferred
        if tensors.ndim == 3:
            tensors = tensors.unsqueeze(0)
        assert tensors.ndim == 4
        image_sizes = [tensors[i].shape[-2:] for i in range(tensors.shape[0])]
        return ImageList(tensors, image_sizes)
    elif isinstance(tensors, (tuple, list)):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

        # TODO Ideally, just remove this and let me model handle arbitrary
        # input sizs
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors),) + max_size
        if isinstance(tensors[0],jt.Var):
            batched_imgs = jt.zeros(batch_shape,dtype='float32')
        else:
            batched_imgs = np.zeros(batch_shape,dtype=np.float32)
        for i in range(len(tensors)):
            img = tensors[i]
            batched_imgs[i,: img.shape[0], : img.shape[1], : img.shape[2]]= img

        image_sizes = [im.shape[-2:] for im in tensors]
        # if isinstance(batched_imgs,np.ndarray):
        #     return batched_imgs,image_sizes
        return ImageList(batched_imgs, image_sizes)
    else:
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))
