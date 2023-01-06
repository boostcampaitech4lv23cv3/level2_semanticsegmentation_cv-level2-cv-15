# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
        dict(type='WandbLoggerHook',
            init_kwargs={
                'entity': 'fullhouse',
                'project': 'Trash Semantic Segmentation',
                'name': 'HJ_uper_convnext-xl_train-all'
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
