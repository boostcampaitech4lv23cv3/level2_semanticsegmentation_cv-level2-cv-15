_base_ = [
    './models/upernet_swin_base.py', './datasets/dataset.py',
    './default_runtime.py', './schedules/schedule.py'
]

load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459-429057bf.pth'