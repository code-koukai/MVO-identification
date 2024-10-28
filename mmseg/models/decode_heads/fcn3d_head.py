import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead
import torch.nn.functional as F

class CustomConvModule3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(CustomConvModule3D, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.SyncBatchNorm(out_channels)
        self.activate = nn.ReLU()


    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activate(x)
        return x


@HEADS.register_module()
class FCN3DHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                
                 **kwargs):
        assert num_convs >= 0
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        # self.dilation = dilation
        print(kwargs.keys())
        super(FCN3DHead, self).__init__(**kwargs)
        

        if num_convs == 0:
            assert self.in_channels == self.channels

        convs = []
        convs.append(
                CustomConvModule3D(self.in_channels, self.channels, kernel_size, padding = kernel_size // 2 ))
        
        for i in range(num_convs - 1):
            convs.append(
                CustomConvModule3D(self.channels, self.channels, kernel_size, padding = kernel_size // 2 ))
            
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = CustomConvModule3D(self.in_channels + self.channels, self.channels, kernel_size, padding = kernel_size // 2 )
            

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))


        # this is for CLASSIFICATION
        output = self.cls_3D(output)
        # # Average pooling across the time dimension
        # output = F.avg_pool3d(output, (30, 1, 1))  # Pooling over the time dimension

        # # Removing the time dimension to get [batch, class_number, h, w]
        # output= output.squeeze(2)


        return output
    
