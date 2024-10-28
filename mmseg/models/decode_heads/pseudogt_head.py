import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead
import torch.nn.functional as F
from mmcv.cnn import (constant_init, kaiming_init)

from mmcv.utils.parrots_wrapper import _BatchNorm


class ConvertToLogits(nn.Module):
    def __init__(self):
        super(ConvertToLogits, self).__init__()

        self.fc = nn.Linear(512, 1)
        
    def forward(self, x):

        x = F.adaptive_avg_pool3d(x, (1, 1, 1))

        x = torch.flatten(x, 1)

        logits = self.fc(x)

        
        return logits

# pseudo_block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
    

class PseudoBranch(nn.Module):
    def __init__(self):
        super(PseudoBranch, self).__init__()
        self.timepooling = nn.AdaptiveMaxPool3d((1, 4, 4))
        self.deconv1 = DecoderBlock(in_channels=512, n_filters=256)
        self.deconv2 = DecoderBlock(in_channels=256, n_filters=128)
        self.dropout = nn.Dropout2d(0.5)
        self.conv_seg = nn.Conv2d(128, 1, kernel_size=1)

        self.dropout_ratio = 0.1
        self.dropout = nn.Dropout2d(self.dropout_ratio)

    def DensePrediction(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output


    def forward(self, x):
        x = self.timepooling(x)
        x = x.squeeze(2)
        # x = [2, 512, 4, 4]
        x = self.deconv1(x)
        x = self.deconv2(x)

        x = self.DensePrediction(x)

        return x

@HEADS.register_module()
class PseudoHead(BaseDecodeHead):
    def __init__(self, num_convs=2, kernel_size=3, concat_input=True, 
                 align_corners=False, num_classes=2, **kwargs):
        super(PseudoHead, self).__init__(num_classes=num_classes, **kwargs)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.align_corners = align_corners
        self.convert_to_logits = ConvertToLogits()

        self.pseudo_branch = PseudoBranch()

    def forward(self, inputs):

        if self.training:
            seg = self.pseudo_branch(inputs)
            cls = self.convert_to_logits(inputs)
            combination = [cls, seg]
            return combination
        else:
            cls = self.convert_to_logits(inputs)
            return cls



