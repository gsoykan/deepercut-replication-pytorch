from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision
import config
from enum import Enum
import numpy as np

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


class DeeperCutBackbone(Enum):
    ResNet50 = 1
    ResNet101 = 2
    ResNet152 = 3


class DeeperCutHead(Enum):
    part_detection = 1
    locref = 2
    intermediate_supervision = 3


def match_sizes_by_adding_padding_for_skip_connection(tensor1, tensor2):
    t1_shape = np.array(tensor1.size()[2:4])
    t2_shape = np.array(tensor2.size()[2:4])
    diff = np.subtract(t1_shape, t2_shape)
    result_t = tensor2
    if (diff == np.array([0, 0])).all():
        pass
    elif (diff == np.array([1, 0])).all():
        result_t = functional.pad(tensor2, pad=(0, 0, 0, 1), mode='replicate')
    elif (diff == np.array([0, 1])).all():
        result_t = functional.pad(tensor2, pad=(0, 1, 0, 0), mode='replicate')
    elif (diff == np.array([1, 1])).all():
        result_t = functional.pad(tensor2, pad=(0, 1, 0, 1), mode='replicate')
    else:
        assert False, "invalid padding for skip connection"
    result_shape = np.array(result_t.size()[2:4])
    assert (result_shape == t1_shape).all(), "Shape mismatch after padding"
    return result_t


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
    def __init__(self, num_joints,
                 backbone=DeeperCutBackbone.ResNet50):
        super(DeeperCut, self).__init__()
        self.enable_skip_connection = config.enable_skip_connections
        self.enable_intermediate_supervision = config.enable_intermediate_supervision
        self.location_refinement = config.location_refinement
        if self.enable_skip_connection:
            self.part_detection_connection_conv = nn.Conv2d(512,
                                                            num_joints,
                                                            kernel_size=(1, 1)
                                                            )
            self.location_refinement_connection_conv = nn.Conv2d(512,
                                                                 num_joints * 2,
                                                                 kernel_size=(1, 1))

        if backbone == DeeperCutBackbone.ResNet50:
            resnet = torchvision.models.resnet50(pretrained=True)
        elif backbone == DeeperCutBackbone.ResNet101:
            resnet = torchvision.models.resnet101(pretrained=True)
        elif backbone == DeeperCutBackbone.ResNet152:
            resnet = torchvision.models.resnet152(pretrained=True)
        else:
            assert False, "unhandled DeerCut backbone error"

        resnet_modules = list(resnet.children())[:-2]
        decrease_resnet_backbone_stride(resnet_modules)

        self.backbone_to_conv3_bank = nn.Sequential(*resnet_modules[:6])
        self.backbone_conv4_block = resnet_modules[6:7][0]
        self.backbone_from_conv4_to_conv5 = nn.Sequential(*resnet_modules[7:])

        self.intermediate_supervision_head = nn.ConvTranspose2d(in_channels=1024,
                                                                out_channels=num_joints,
                                                                kernel_size=(3, 3),
                                                                stride=2,
                                                                padding=1,
                                                                output_padding=1
                                                                )
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

    def forward(self, x):
        result_dict = {}
        backbone_until_conv3_result = self.backbone_to_conv3_bank(x)
        backbone_conv4_result = self.backbone_conv4_block(backbone_until_conv3_result)
        backbone_result = self.backbone_from_conv4_to_conv5(backbone_conv4_result)
        part_detection_result = self.part_detection_head(backbone_result)

        if self.enable_skip_connection:
            part_detection_skip_result = self.part_detection_connection_conv(backbone_until_conv3_result)
            part_detection_skip_result = match_sizes_by_adding_padding_for_skip_connection(part_detection_result,
                                                                                           part_detection_skip_result)
            part_detection_result = part_detection_result + part_detection_skip_result

        result_dict[DeeperCutHead.part_detection] = part_detection_result

        if self.location_refinement:
            locref_result = self.location_refinement_head(backbone_result)

            if self.enable_skip_connection:
                locref_skip_result = self.location_refinement_connection_conv(backbone_until_conv3_result)
                locref_skip_result = match_sizes_by_adding_padding_for_skip_connection(locref_result,
                                                                                       locref_skip_result)
                locref_result = locref_result + locref_skip_result

            result_dict[DeeperCutHead.locref] = locref_result

        if self.enable_intermediate_supervision:
            intermediate_supervision_result = self.intermediate_supervision_head(backbone_conv4_result)
            result_dict[DeeperCutHead.intermediate_supervision] = intermediate_supervision_result

        return result_dict

    def freeze_bn(self):
        """Freeze BatchNorm layers mean and variance"""
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
