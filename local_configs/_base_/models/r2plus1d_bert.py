# model settings  
norm_cfg = dict(type='SyncBN', requires_grad=True)  
model = dict(  
    type='EncoderDecoder',  
    pretrained=None,  
    backbone=dict(  
        type='r2plus1d_bert_Net'),  
    decode_head=dict(  
        sampler=dict(type='OHEMPixelSampler', thresh=0.5, min_kept=100000),  
        type='UnchangedHead',  
        in_channels=64,  
        in_index=3,  
        channels=64,  
        num_convs=1,  
        dropout_ratio=0.1,  
        num_classes=4,  
        norm_cfg=norm_cfg,  
        align_corners=False,  
        decoder_params=dict(embed_dim=256),  
        ignore_index=0,  
        loss_decode=dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0, avg_non_ignore=True),  
    ),  
    train_cfg=dict(),  
    test_cfg=dict(mode='slide', crop_size=256, stride=170))