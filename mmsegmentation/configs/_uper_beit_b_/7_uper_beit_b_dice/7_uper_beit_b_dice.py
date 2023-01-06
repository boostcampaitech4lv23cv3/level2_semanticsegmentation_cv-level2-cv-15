_base_ = [
    './_base_/models/upernet_beit.py', 
    # './_base_/datasets/custom_dataset.py',
    './_base_/datasets/custom_dataset.py',
    '../../_base_/default_runtime.py', './_base_/schedules/schedule_160k.py'
]


load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/beit/upernet_beit-base_8x2_640x640_160k_ade20k/upernet_beit-base_8x2_640x640_160k_ade20k-eead221d.pth'

model = dict(
    pretrained='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22k.pth',
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426)))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=3e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
)
    # by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)

# data = dict(samples_per_gpu=2)
optimizer_config = dict(
    type='GradientCumulativeFp16OptimizerHook', cumulative_iters=2)

fp16 = dict()
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
            init_kwargs={
                'project': 'Trash Semantic Segmentation',
                'entity': 'fullhouse',
                'name': 'SY_7_uper_beit_b_dice'
            },
            )
    ])

log_level = 'INFO'
workflow = [('train', 1),('val',1)]
runner = dict(type='EpochBasedRunner', max_epochs=22)
checkpoint_config = dict(interval=8)
evaluation = dict(metric='mIoU', save_best='mIoU')

# evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
# evaluation = dict(metric='mIoU', save_best='mIoU')
# checkpoint_config = dict(by_epoch=False, interval=16000)
# yapf:enable

