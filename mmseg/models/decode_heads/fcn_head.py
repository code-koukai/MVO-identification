import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class FCNHead(BaseDecodeHead):
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
                #  dilation=[-9,-6,-3],
                 **kwargs):
        assert num_convs >= 0
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        # self.dilation = dilation
        print(kwargs.keys())
        super(FCNHead, self).__init__(**kwargs)
        kwargs.pop('dilation', None)

        if num_convs == 0:
            assert self.in_channels == self.channels

        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""

        x = self._transform_inputs(inputs)
        output = self.convs(x)

        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        #print("output2222",output.shape)
        # torch.Size([2, 64, 64, 64])


        output = self.cls_2D(output)

        
        return output
'''

 torch.Size([2, 64, 64, 64])
output torch.Size([2, 64, 64, 64])
output2222output2222  torch.Size([2, 64, 64, 64])torch.Size([2, 64, 64, 64])

output2222 torch.Size([2, 64, 64, 64])
torch.Size([2, 64, 64, 64])
torch.Size([2, 64, 64, 64])
torch.Size([2, 64, 64, 64])
output torch.Size([2, 64, 64, 64])
output torch.Size([2, 64, 64, 64])
output torch.Size([2, 64, 64, 64])
output2222output2222  torch.Size([2, 64, 64, 64])
torch.Size([2, 64, 64, 64])'''