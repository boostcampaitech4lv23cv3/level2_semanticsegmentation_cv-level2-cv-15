_base_ = [
    './_base_/models/deeplabv3_r50-d8.py', 
    './_base_/datasets/custom_dataset.py',
    './_base_/default_runtime.py', 
    './_base_/schedules/scheduler.py'
]

# evaluation = dict(interval=1, metric='mIoU', classwise=True, save_best='mIoU_best')