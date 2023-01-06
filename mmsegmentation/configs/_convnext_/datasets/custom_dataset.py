dataset_type = 'CustomDataset'
data_root = '/opt/ml/input/data/mmseg/'
img_dir = '/opt/ml/input/data/mmseg/img_dir/'
ann_dir = '/opt/ml/input/data/mmseg/ann_dir/'


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

classes = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

palette = [
    [0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64], [128, 0, 0], [64, 0, 128],
    [64, 0, 192], [192, 128, 64], [192, 192, 128], [64, 64, 128], [128, 0, 192]
]

albu_transforms = [
    dict(type='HorizontalFlip', p=0.3),
    dict(type='OneOf', transforms=[
        dict(type='Blur', p=1.0),
        dict(type='MotionBlur', p=1.0),
        dict(type='GaussianBlur', p=1.0),
        dict(type='MedianBlur', p=1.0),
        dict(type='Sharpen', p=1.0),
        dict(type='Emboss', p=1.0)
    ], p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(type='ISONoise', p=1.0),
            dict(type='ColorJitter', brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=1.0),
            dict(type='RandomGamma', p=1.0),
            dict(type='HueSaturationValue', p=1.0),
            dict(type='ChannelShuffle', p=1.0),
            dict(type='RGBShift', p=1.0),
            dict(type='CLAHE', clip_limit=(1, 10), p=1.0),
            dict(type='RandomBrightnessContrast', p=1.0), 
    ], p=0.3),
]

crop_size = (336, 336)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='Albu',
        transforms=albu_transforms,
        keymap=dict(img="image", gt_semantic_seg="mask"),
        update_pad_shape=False,),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_dir=ann_dir + 'K-fold_train1',
        img_dir=img_dir + 'K-fold_train1',
        classes=classes,
        palette=palette,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_dir=ann_dir + 'K-fold_val1',
        img_dir=img_dir + 'K-fold_val1',
        classes=classes,
        palette=palette,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        img_dir=img_dir+'test',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))