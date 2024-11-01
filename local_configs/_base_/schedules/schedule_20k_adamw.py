# optimizer
optimizer = dict(type='AdamW',  lr=0.00001, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=1500)
evaluation = dict(interval=50, metric='mIoU')
