from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from scipy.io import loadmat
import config
import dataset.mpii

"""
     deepercut_head = DeeperCutHead(;
            part_detection_head = Deconv(
                3,
                3,
                global_num_joints,
                2048;
                padding = 1,
                stride = 2,
                tag = "part_detect_deconv",
            ),
            loc_ref_head = Deconv(
                3,
                3,
                global_num_joints * 2,
                2048;
                padding = 1,
                stride = 2,
                tag = "loc_ref_deconv",
            ),
            is_loc_ref_enabled = true,
        )
"""


def decrease_resnet_backbone_stride(resnet_modules):
    bottlenecks = list(resnet_modules[-1].children())
    first_bottleneck = bottlenecks[0]
    first_bottleneck.conv2.stride = 1
    first_bottleneck_downsample_conv = list(first_bottleneck.downsample.children())[0]
    first_bottleneck_downsample_conv.stride = 1
    for i in range(1, len(bottlenecks)):
        bottleneck = bottlenecks[i]
        bottleneck.conv2.dilation = 2
        bottleneck.conv2.padding = 2


class DeeperCut(torch.nn.Module):
    def __init__(self, num_joints):
        super(DeeperCut, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet_modules = list(resnet.children())[:-2]
        decrease_resnet_backbone_stride(resnet_modules)
        self.backbone = nn.Sequential(*resnet_modules)
        self.part_detection_head = nn.ConvTranspose2d(in_channels=2048,
                                                      out_channels=num_joints,
                                                      kernel_size=(3, 3),
                                                      stride=2,
                                                      padding=1,
                                                      output_padding=1
                                                      )
        # TODO: implement location refinement head

    def forward(self, x):
        backbone_result = self.backbone(x)
        part_detection_result = self.part_detection_head(backbone_result)
        return part_detection_result


if __name__ == '__main__':
    dc = DeeperCut(config.num_joints);
