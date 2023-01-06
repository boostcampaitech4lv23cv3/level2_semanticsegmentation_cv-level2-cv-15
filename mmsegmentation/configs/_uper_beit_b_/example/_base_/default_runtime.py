checkpoint_config = dict(interval=3, max_keep_ckpts = 1)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
            init_kwargs={
                'project': 'Trash Semantic Segmentation',
                'entity': 'fullhouse',
                'name': 'sy_test1'
            },
            )
    ])


# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1),('val', 1)]
cudnn_benchmark = True
# checkpoint_config = dict(interval=3, max_keep_ckpts = 1)
