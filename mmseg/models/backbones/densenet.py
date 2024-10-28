import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (UPSAMPLE_LAYERS, ConvModule, build_activation_layer,
                      build_norm_layer, constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import UpConvBlock

import torch
import torch.utils.checkpoint as cp


import torchvision.models.video as models
import numpy as np

import torchvision.models as models


import torch
import torch.nn as nn
import torchvision.models as models

# class ModifiedResNet(nn.Module):
#     def __init__(self):
#         super(ModifiedResNet, self).__init__()
    
#         self.resnet = models.resnet50(pretrained=True)

#         original_conv1 = self.resnet.conv1
#         self.resnet.conv1 = nn.Conv2d(30, original_conv1.out_channels,
#                                       kernel_size=original_conv1.kernel_size,
#                                       stride=original_conv1.stride,
#                                       padding=original_conv1.padding,
#                                       bias=False)

       
#         num_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_features, 1)

#     def forward(self, x):
#         return self.resnet(x)



class ModifiedDenseNet(nn.Module):
    def __init__(self):
        super(ModifiedDenseNet, self).__init__()

        self.densenet = models.densenet121(pretrained=True)
        
        original_conv0 = self.densenet.features.conv0
        self.densenet.features.conv0 = nn.Conv2d(30, original_conv0.out_channels,
                                                 kernel_size=original_conv0.kernel_size,
                                                 stride=original_conv0.stride,
                                                 padding=original_conv0.padding,
                                                 bias=False)

        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.densenet(x)


@BACKBONES.register_module()
class DenseNet(nn.Module):
 
    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 num_stages=5,
                 strides=(1, 1, 1, 1, 1),
                 enc_num_convs=(2, 2, 2, 2, 2),
                 dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True),
                 enc_dilations=(1, 1, 1, 1, 1),
                 dec_dilations=(1, 1, 1, 1),
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='DeconvModule'),
                 norm_eval=False,
                 dcn=None,
                 plugins=None):
        super(DenseNet, self).__init__()

        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.norm_eval = norm_eval

        self.densenet = ModifiedDenseNet()

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            if self.training:
                # Squeeze operation for training phase
                x = x[:, :, 0, :, :].squeeze(2)
            # Additional operations for tensors can be added here if needed
        elif isinstance(x, list):
            # Operations to perform if x is a list
            x = [tensor[:, 0:1, :, :] for tensor in x]
            x = torch.cat(x, dim=1)
        else:
            # Optional: Handle unexpected types
            raise TypeError(f"Unexpected type for 'x': {type(x)}")

        return self.densenet(x)
    
    def init_weights(self, pretrained=None):

        if isinstance(pretrained, str):
            print("no pretrained model")
            assert pretrained.endswith('.pth'), 'no pretrained model'
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

