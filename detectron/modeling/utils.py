# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Miscellaneous utility functions
"""

import jittor as jt 
from jittor import nn,Module,init

def cat(tensors, dim=0):
   
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return jt.contrib.concat(tensors, dim)
