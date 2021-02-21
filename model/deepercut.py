from __future__ import print_function, division

import torch
import torch.nn as nn
import torchvision
import config
from enum import Enum

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


class DeeperCutHead(Enum):
    part_detection = 1
    locref = 2


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
        self.location_refinement_head = nn.ConvTranspose2d(in_channels=2048,
                                                           out_channels=2 * num_joints,
                                                           kernel_size=(3, 3),
                                                           stride=2,
                                                           padding=1,
                                                           output_padding=1
                                                           )
        # TODO: implement location refinement head

    def forward(self, x):
        result_dict = {}
        backbone_result = self.backbone(x)
        part_detection_result = self.part_detection_head(backbone_result)
        result_dict[DeeperCutHead.part_detection] = part_detection_result
        if config.location_refinement:
            loc_ref_result = self.location_refinement_head(backbone_result)
            result_dict[DeeperCutHead.locref] = loc_ref_result
        return result_dict

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
"""

-- MULTIPLE LOSS SCRIPT

b = nn.MSELoss()
a = nn.CrossEntropyLoss()

loss_a = a(output_x, x_labels)
loss_b = b(output_y, y_labels)

loss = loss_a + loss_b

loss.backward()
"""

if __name__ == '__main__':
    dc = DeeperCut(config.num_joints);
