import torch  
import torch.nn as nn  
import torch.utils.checkpoint as cp  
import torch.nn.functional as F  
from mmcv.cnn import (UPSAMPLE_LAYERS, ConvModule, build_activation_layer,  
                      build_norm_layer, constant_init, kaiming_init)  
from mmcv.runner import load_checkpoint  
from mmcv.utils.parrots_wrapper import _BatchNorm  
from mmseg.utils import get_root_logger  
from ..builder import BACKBONES, HEADS  
from ..utils import UpConvBlock  
import torchvision.models.video as video_models  
from torchvision.models.video import r2plus1d_18  
import torchvision.models as models


class ModifiedR2Plus1D(nn.Module):
    def __init__(self):
        super(ModifiedR2Plus1D,self).__init__()

        original_model = r2plus1d_18(pretrained=True)
        self.features = nn.Sequential(*list(original_model.children())[:-2])

    def forward(self, x):

        x = self.features(x)
        return x

@BACKBONES.register_module()
class R2plus1DwithoutFCNet(nn.Module):
 
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
        super(R2plus1DwithoutFCNet, self).__init__()

        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.norm_eval = norm_eval

        self.feature = ModifiedR2Plus1D()

    

    
    def forward(self, x):


        if isinstance(x, list):
            x = [t.unsqueeze(2) for t in x]
           
            x = torch.cat(x, dim=2)
            
        elif torch.is_tensor(x):
            x = x.permute(0, 2, 1, 3, 4)

        x = self.feature(x)

        return x

        
    
    def init_weights(self, pretrained=None):
        pass

