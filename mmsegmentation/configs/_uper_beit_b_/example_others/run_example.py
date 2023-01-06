_base_ = [
    './_base_/datasets/dataset.py',
    './_base_/models/pspnet_r50-d8.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_1x.py'
]

# evaluation = dict(interval=1, metric='mIoU', classwise=True, save_best='mIoU_best')