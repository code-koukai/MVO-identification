import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead
import torch.nn.functional as F
from mmcv.cnn import (constant_init, kaiming_init)

from mmcv.utils.parrots_wrapper import _BatchNorm
from .BERT.bert import BERT, BERT2, BERT3, BERT4, BERT5, BERT6

class BertBranch(BaseDecodeHead):
    def __init__(self, num_convs=2, kernel_size=3, concat_input=True, 
                 align_corners=False, num_classes=2, **kwargs):
        super(BertBranch, self).__init__(num_classes=num_classes, **kwargs)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.align_corners = align_corners

        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=2
        self.length=30
        self.dp = nn.Dropout(p=0.8)


        self.bert = BERT5(self.hidden_size,4, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)

        
        self.fc_action = nn.Linear(self.hidden_size, 1)

        #self.original_model = r2plus1d_18(pretrained=True)
        #self.r2plus1d = ModifiedR2Plus1D(self.original_model)
        # self.r2plus1d = ModifiedR2Plus1DFC()

        self.avgpool = nn.AvgPool3d((1, 4, 4), stride=1)
        
        self.avgpool1 = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 1) 
    def forward(self, inputs):
       
        x = self.avgpool(inputs)
        #print("x after avgpool",x.shape)
        
        x = x.view(x.size(0), self.hidden_size, 4)
        x = x.transpose(1,2)


        #print("x after transpose",x.shape)
        #[4,4,512]'

        input_vectors=x
        norm = input_vectors.norm(p=2, dim = -1, keepdim=True)
        
        input_vectors = input_vectors.div(norm)
        output , maskSample = self.bert(x)
        
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        norm = sequenceOut.norm(p=2, dim = -1, keepdim=True)
        sequenceOut = sequenceOut.div(norm)
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x

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
class BertPseudoHead(BaseDecodeHead):
    def __init__(self, num_convs=2, kernel_size=3, concat_input=True, 
                 align_corners=False, num_classes=2, **kwargs):
        super(BertPseudoHead, self).__init__(num_classes=num_classes, **kwargs)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.align_corners = align_corners

        self.pseudo_branch = PseudoBranch()
        self.bert_branch = BertBranch(num_convs=num_convs, kernel_size=kernel_size, concat_input=concat_input, 
                                      align_corners=align_corners, num_classes=num_classes, **kwargs)

    def forward(self, inputs):

        if self.training:
            seg = self.pseudo_branch(inputs)
            cls = self.bert_branch(inputs)
            combination = [cls, seg]
            return combination
        else:
            cls = self.bert_branch(inputs)
            return cls



