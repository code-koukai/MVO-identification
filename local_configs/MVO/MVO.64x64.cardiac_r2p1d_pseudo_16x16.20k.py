_base_ = [  
    '../_base_/models/r2plus1d_bert.py',  
   # '../_base_/datasets/',   #Please enter your own dataset here.
    '../_base_/default_runtime.py',  
    '../_base_/schedules/schedule_10k_adamw.py'  
]  

# model settings  
norm_cfg = dict(type='SyncBN', requires_grad=True)  
find_unused_parameters = True  
model = dict(  
    type='EncoderDecoder',  
    backbone=dict(  
        type='R2plus1DwithoutFCNet'),  
    decode_head=dict(  
        type='PseudoHead',  
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
    test_cfg=dict(mode='whole'))  

# optimizer  
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)  

lr_config = dict(_delete_=True, policy='poly',  
                 warmup='linear',  
                 warmup_iters=1500,  
                 warmup_ratio=1e-6,  
                 power=1.0, min_lr=0.0, by_epoch=False)  

data = dict(samples_per_gpu=4)  
evaluation = dict(interval=50, metric='mIoU')