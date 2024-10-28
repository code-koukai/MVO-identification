from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmcv.runner import auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy
import torch.nn.functional as F
from mmcv.utils import print_log

class BaseDecodeHead(nn.Module, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 decoder_params=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False):
        super(BaseDecodeHead, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        self.cls2Dconv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.cls2Dconv2 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.cls2Dconv3 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        
        self.cls2Dpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.cls2Dpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.cls2Dpool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.cls2Dfc = nn.Linear(32768, 1)




        self.global_avg_pool =  nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(in_channels, 1)  

        self.clsconv1 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.clsconv2 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.clsconv3 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)

        self.clspool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.clspool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.clspool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.clsfc = nn.Linear(32768, 1)


        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.conv_seg3D = nn.Conv3d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
            self.dropout3D = nn.Dropout3d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self):
        """Initialize weights of classification layer."""
        normal_init(self.conv_seg, mean=0, std=0.01)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        gt_semantic_seg = gt_semantic_seg[:, 0]

        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output
    
    def cls_seg_3D(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout3D(feat)
        output = self.conv_seg3D(feat) 
        return output
    
 
    def cls_2D(self, feat):
        if self.dropout is not None:
            feat = self.dropout(feat)
        x = F.relu(self.cls2Dconv1(feat))
        x = self.cls2Dpool1(x)
        x = F.relu(self.cls2Dconv2(x))
        x = self.cls2Dpool2(x)
        x = F.relu(self.cls2Dconv3(x))
        x = self.cls2Dpool3(x)

        x = x.view(x.size(0), -1)
        
        output = self.cls2Dfc(x)
        return output
    

    def cls_3D(self, feat):
        if self.dropout is not None:
            feat = self.dropout3D(feat)
        x = F.relu(self.clsconv1(feat))
        x = self.clspool1(x)
        x = F.relu(self.clsconv2(x))
        x = self.clspool2(x)
        x = F.relu(self.clsconv3(x))
        x = self.clspool3(x)

        x = x.view(x.size(0), -1)

        output = self.clsfc(x)
        return output
    


    def weighted_binary_cross_entropy(self,output, target, weights=None, epsilon=1e-7):
        if weights is not None:
            assert len(weights) == 2

            loss = weights[1] * (target * torch.log(output + epsilon)) + \
                weights[0] * ((1 - target) * torch.log(1 - output+epsilon))
        else:
            loss = target * torch.log(output + epsilon) + (1 - target) * torch.log(1 - output + epsilon)

        return torch.neg(torch.mean(loss))
    
    def MSE_loss(self, predict, gt):
        mse_loss_fn = nn.MSELoss()

        mse_loss = mse_loss_fn(predict, gt)
        return mse_loss


    weights = [340/ 680,340/ 680]


    def focal_loss(self, output, target, alpha=0.25, gamma=2.0):
        BCE_loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = alpha * (1 - pt) ** gamma * BCE_loss

        return torch.mean(F_loss)
    
    def compute_accuracy(self, pred, target):
        """
        Compute the accuracy of predictions for a binary classification task.

        Args:
        pred (torch.Tensor): The predictions from the model. Shape: [batch_size, 1]
        target (torch.Tensor): The actual target labels. Shape: [batch_size, 1]

        Returns:
        float: The accuracy as a percentage.
        """
        pred_classes = (pred >= 0.5).float()
        
        correct_predictions = torch.eq(pred_classes, target.float())
        
        accuracy = torch.mean(correct_predictions.float()) * 100
        return accuracy
    
    def process_labels(self,seg_label):

        has_class_2 = (seg_label == 2)
        
        has_class_2 = has_class_2.any(dim=-1).any(dim=-1).float()
     

        has_class_2 = has_class_2.view(-1, 1)

        return has_class_2
    

    def pseudo_gt_generation(self, seg_label):
   
        
        output = torch.zeros((seg_label.shape[0], seg_label.shape[1], seg_label.shape[2]//4, seg_label.shape[3]//4)).to(seg_label.device)


        for batch_idx in range(seg_label.shape[0]):

            img = seg_label[batch_idx, 0]
            blocks = img.unfold(0, 4, 4).unfold(1, 4, 4)

            for i in range(blocks.shape[0]):
                for j in range(blocks.shape[1]):
                    block = blocks[i, j]
                    pixels_value_2 = (block == 2).sum(dtype=torch.float32)
                    ratio = pixels_value_2 / 16.0
                    
                    output[batch_idx, 0, i, j] = ratio

        return output




    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""

        loss = dict()
        

        if type(seg_logit) == list:
          

            cls, seg = seg_logit

            
            pseudo_seg_label = self.pseudo_gt_generation(seg_label)

                
            pseudo_seg_label = pseudo_seg_label.squeeze(1)
            seg = seg.squeeze(1)

            mse_loss = self.MSE_loss(seg, pseudo_seg_label)



            seg_label_cls = self.process_labels(seg_label)

            
            cls = torch.sigmoid(cls)

            weighted_bce_loss = self.weighted_binary_cross_entropy(cls, seg_label_cls.float(), self.weights)


            loss_cls = weighted_bce_loss 
            loss_seg = mse_loss
            loss['loss_seg'] = loss_cls + 0.05*loss_seg

            acc_cls = self.compute_accuracy(cls, seg_label_cls).unsqueeze(0)
            loss['acc_seg'] = acc_cls
        else:

            
            if len(seg_logit.shape) == 4:
               
                seg_logit = resize(
                    input=seg_logit,
                    size=seg_label.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)


                if self.sampler is not None:
                    seg_weight = self.sampler.sample(seg_logit, seg_label)
                else:
                    seg_weight = None
                seg_label_seg = seg_label.squeeze(1)


                loss['loss_seg'] = self.loss_decode(
                    seg_logit,
                    seg_label_seg,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
                loss['acc_seg'] = accuracy(seg_logit, seg_label_seg)



            if len(seg_logit.shape) == 2:
                
                seg_label_cls = self.process_labels(seg_label)

                seg_logit = torch.sigmoid(seg_logit)
                weighted_bce_loss = self.weighted_binary_cross_entropy(seg_logit, seg_label_cls.float(), self.weights)

                loss['loss_seg'] = weighted_bce_loss
                loss['acc_seg'] = self.compute_accuracy(seg_logit, seg_label_cls).unsqueeze(0)

                print_log(f"loss_cls: {loss['loss_seg']}, acc_cls: {loss['acc_seg']}", logger=None)

        return loss



class BaseDecodeHead_clips(nn.Module, metaclass=ABCMeta):
    """Base class for BaseDecodeHead_clips.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 decoder_params=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 num_clips=5,
                 hypercorre=False,
                 cityscape=False,
                 backbone='b1'):
        super(BaseDecodeHead_clips, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.num_clips=num_clips

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False
        self.hypercorre=hypercorre
        self.atten_loss=False
        self.self_ensemble2=False
        self.cityscape=cityscape
        self.backbone=backbone

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self):
        """Initialize weights of classification layer."""
        normal_init(self.conv_seg, mean=0, std=0.01)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg,batch_size, num_clips):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_logits = self.forward(inputs,batch_size, num_clips)
        

        gt_semantic_seg = gt_semantic_seg[:, 0:1, :, :, :]
       
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, batch_size=None, num_clips=None):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs, batch_size, num_clips)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def _construct_ideal_affinity_matrix(self, label, label_size):
        assert label.dim()==5
        B,num_clips,c,h_label,w_label=label.shape
        assert c==1
        label=label.reshape(B*num_clips,c,h_label,w_label)
        scaled_labels = F.interpolate(
            label.float(), size=label_size, mode="nearest")
        scaled_labels = scaled_labels.squeeze_().long()
        scaled_labels[scaled_labels == 255] = self.num_classes

        one_hot_labels = F.one_hot(scaled_labels, self.num_classes + 1)
        one_hot_labels = one_hot_labels.view(
            B, num_clips, -1, self.num_classes + 1).float()
        one_hot_labels_lastframe=one_hot_labels[:,-1:]
        one_hot_labels_reference=one_hot_labels[:,:-1]

        ideal_affinity_matrix=torch.matmul(one_hot_labels_lastframe,
                                           one_hot_labels_reference.transpose(-2, -1))
        assert ideal_affinity_matrix.dim()==4
        return ideal_affinity_matrix.reshape(B*(num_clips-1), ideal_affinity_matrix.shape[-2], ideal_affinity_matrix.shape[-1])
    


    def focal_loss(output, target, alpha=0.25, gamma=2.0):
        BCE_loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = alpha * (1 - pt) ** gamma * BCE_loss

        return torch.mean(F_loss)

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""


        assert seg_logit.dim()==5 and seg_label.dim()==5

        loss = dict()
        


        if self.hypercorre and self.cityscape:



            assert seg_logit.shape[1]==2*seg_label.shape[1]
   
            seg_logit_ori=seg_logit[:,num_clips-1:num_clips]

            batch_size, _, _, h ,w=seg_logit_ori.shape
            seg_logit_ori=seg_logit_ori.reshape(batch_size,-1,h,w)
            seg_logit_lastframe=seg_logit[:,num_clips:].reshape(batch_size*(num_clips),-1,h,w)

            batch_size, num_clips, _, h ,w=seg_label.shape
            seg_label_ori=seg_label[:,-1]
            seg_label_lastframe=seg_label[:,-1:].expand(batch_size,num_clips,1,h,w).reshape(batch_size*(num_clips),1,h,w)
        elif self.hypercorre:

            if self.self_ensemble2 and seg_logit.shape[1]==2*seg_label.shape[1]:

                assert seg_logit.shape[1]==2*seg_label.shape[1]
                num_clips=seg_label.shape[1]

                seg_logit_ori=seg_logit[:,:num_clips]
                batch_size, _, _, h ,w=seg_logit_ori.shape
                seg_logit_ori=seg_logit_ori.reshape(batch_size*(num_clips),-1,h,w)
                seg_logit_lastframe=seg_logit[:,num_clips:].reshape(batch_size*(num_clips),-1,h,w)

                batch_size, num_clips, chan, h ,w=seg_label.shape
                assert chan==1
                seg_label_ori=seg_label.reshape(batch_size*(num_clips),1,h,w)
                seg_label_lastframe=seg_label[:,-1:].expand(batch_size,num_clips,1,h,w).reshape(batch_size*(num_clips),1,h,w)
            else:
                assert False, "parameters not correct"            

        seg_logit_ori = resize(
            input=seg_logit_ori,
            size=seg_label.shape[3:],
            mode='bilinear',
            align_corners=self.align_corners)

        seg_logit_lastframe = resize(
            input=seg_logit_lastframe,
            size=seg_label.shape[3:],
            mode='bilinear',
            align_corners=self.align_corners)

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None

        seg_label_ori = seg_label_ori.squeeze(1)
        seg_label_lastframe = seg_label_lastframe.squeeze(1)

        loss['loss_seg'] = 0.5*self.loss_decode(
            seg_logit_ori,
            seg_label_ori,
            weight=seg_weight,
            ignore_index=self.ignore_index)+self.loss_decode(
            seg_logit_lastframe,
            seg_label_lastframe,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit_ori, seg_label_ori)


        return loss


