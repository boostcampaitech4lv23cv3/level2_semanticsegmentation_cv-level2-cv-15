_base_ = [
    './models/upernet_swin.py', './datasets/custom_dataset.py',
    './default_runtime.py', './schedules/schedule.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=11),
    auxiliary_head=dict(in_channels=384, num_classes=11))

# upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K