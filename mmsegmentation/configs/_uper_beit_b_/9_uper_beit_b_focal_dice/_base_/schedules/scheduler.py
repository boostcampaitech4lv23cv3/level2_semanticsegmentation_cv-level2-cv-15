# # optimizer
# optimizer_config = dict(grad_clip=None)
# optimizer = dict(
#     type='AdamW',
#     lr=0.0001,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=250,
#     warmup_ratio=1.0 / 10,
#     min_lr_ratio=1e-5)

# runner = dict(type='EpochBasedRunner', max_epochs=30)
# # checkpoint_config = dict(epoch=10)

# optimizer
optimizer = dict(type='AdamW',
                lr=0.00006,
                weight_decay=0.0001,
                paramwise_cfg=dict(
                custom_keys={
                'absolute_pos_embed': dict(decay_mult=0.),
                'relative_position_bias_table': dict(decay_mult=0.),
                'norm': dict(decay_mult=0.)})
                )

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# which may diverge with large learning rates, and 35 is just a empirical value.

# learning policy
lr_config = dict(
        policy='CosineAnnealing',
        warmup='linear',
        warmup_iters=500, 
        warmup_ratio=0.01,
        min_lr=1e-05,
    )
runner = dict(type='EpochBasedRunner', max_epochs=30)