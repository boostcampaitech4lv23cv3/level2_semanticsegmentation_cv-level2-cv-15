# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
        dict(type='WandbLoggerHook',
            init_kwargs={
                'entity': 'fullhouse',
                'project': 'Trash Semantic Segmentation',
                'name': 'DH_swin-b_aug_test'
            }
        )
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True
