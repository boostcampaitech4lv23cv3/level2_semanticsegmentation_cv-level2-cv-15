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
                'name': 'DH_test_all'
            }
        )
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = '/opt/ml/level2_semanticsegmentation_cv-level2-cv-15/mmsegmentation/work_dirs/swin_large_del_all/best_mIoU_epoch_90.pth'
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True
