auto_scale_lr = dict(base_batch_size=16, enable=False)
backend = 'pillow'
backend_args = None
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'diffusiondet',
    ])
data_root = './dentex-4'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=200, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=40, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=True, type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)
metainfo = dict(
    classes=(
        'Teeth',
        'Caries',
        'Deep Caries',
        'Impacted',
        'Periapical Lesion',
    ))
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        criterion=dict(
            assigner=dict(
                candidate_topk=5,
                center_radius=2.5,
                match_costs=[
                    dict(
                        alpha=0.25,
                        eps=1e-08,
                        gamma=2.0,
                        type='FocalLossCost',
                        weight=2.0),
                    dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                    dict(iou_mode='giou', type='IoUCost', weight=2.0),
                ],
                type='DiffusionDetMatcher'),
            loss_bbox=dict(loss_weight=5.0, reduction='sum', type='L1Loss'),
            loss_cls=dict(
                alpha=0.25,
                gamma=2.0,
                loss_weight=2.0,
                reduction='sum',
                type='FocalLoss',
                use_sigmoid=True),
            loss_giou=dict(loss_weight=2.0, reduction='sum', type='GIoULoss'),
            num_classes=5,
            type='DiffusionDetCriterion'),
        ddim_sampling_eta=1.0,
        deep_supervision=True,
        feat_channels=256,
        num_classes=5,
        num_heads=6,
        num_proposals=500,
        prior_prob=0.01,
        roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=2, type='RoIAlign'),
            type='SingleRoIExtractor'),
        sampling_timesteps=1,
        single_head=dict(
            act_cfg=dict(inplace=True, type='ReLU'),
            dim_feedforward=2048,
            dropout=0.0,
            dynamic_conv=dict(dynamic_dim=64, dynamic_num=2),
            num_cls_convs=1,
            num_heads=8,
            num_reg_convs=3,
            type='SingleDiffusionDetHead'),
        snr_scale=2.0,
        type='DynamicDiffusionDetHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=4,
        out_channels=256,
        type='FPN'),
    test_cfg=dict(
        min_bbox_size=0,
        nms=dict(iou_threshold=0.5, type='nms'),
        score_thr=0.5,
        use_nms=True),
    type='DiffusionDet')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(lr=2.5e-05, type='AdamW', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.01, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=False,
        end=450000,
        gamma=0.1,
        milestones=[
            350000,
            420000,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='test/_annotations.coco.json',
        backend_args=None,
        data_prefix=dict(img='test/'),
        data_root='./dentex-4',
        metainfo=dict(
            classes=(
                'Teeth',
                'Caries',
                'Deep Caries',
                'Impacted',
                'Periapical Lesion',
            )),
        pipeline=[
            dict(
                backend_args=None,
                imdecode_backend='pillow',
                type='LoadImageFromFile'),
            dict(
                backend='pillow',
                keep_ratio=True,
                scale=(
                    1333,
                    800,
                ),
                type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='./dentex-4/test/_annotations.coco.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(
        backend_args=None, imdecode_backend='pillow',
        type='LoadImageFromFile'),
    dict(
        backend='pillow', keep_ratio=True, scale=(
            1333,
            800,
        ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_iters=450000, type='IterBasedTrainLoop', val_interval=200)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=4,
    dataset=dict(
        ann_file='train/_annotations.coco.json',
        backend_args=None,
        data_prefix=dict(img='train/'),
        data_root='./dentex-4',
        filter_cfg=dict(filter_empty_gt=False, min_size=1e-05),
        metainfo=dict(
            classes=(
                'Teeth',
                'Caries',
                'Deep Caries',
                'Impacted',
                'Periapical Lesion',
            )),
        pipeline=[
            dict(
                backend_args=None,
                imdecode_backend='pillow',
                type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                transforms=[
                    [
                        dict(
                            backend='pillow',
                            keep_ratio=True,
                            scales=[
                                (
                                    480,
                                    1333,
                                ),
                                (
                                    512,
                                    1333,
                                ),
                                (
                                    544,
                                    1333,
                                ),
                                (
                                    576,
                                    1333,
                                ),
                                (
                                    608,
                                    1333,
                                ),
                                (
                                    640,
                                    1333,
                                ),
                                (
                                    672,
                                    1333,
                                ),
                                (
                                    704,
                                    1333,
                                ),
                                (
                                    736,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    800,
                                    1333,
                                ),
                            ],
                            type='RandomChoiceResize'),
                    ],
                    [
                        dict(
                            backend='pillow',
                            keep_ratio=True,
                            scales=[
                                (
                                    400,
                                    1333,
                                ),
                                (
                                    500,
                                    1333,
                                ),
                                (
                                    600,
                                    1333,
                                ),
                            ],
                            type='RandomChoiceResize'),
                        dict(
                            allow_negative_crop=True,
                            crop_size=(
                                384,
                                600,
                            ),
                            crop_type='absolute_range',
                            type='RandomCrop'),
                        dict(
                            backend='pillow',
                            keep_ratio=True,
                            scales=[
                                (
                                    480,
                                    1333,
                                ),
                                (
                                    512,
                                    1333,
                                ),
                                (
                                    544,
                                    1333,
                                ),
                                (
                                    576,
                                    1333,
                                ),
                                (
                                    608,
                                    1333,
                                ),
                                (
                                    640,
                                    1333,
                                ),
                                (
                                    672,
                                    1333,
                                ),
                                (
                                    704,
                                    1333,
                                ),
                                (
                                    736,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    800,
                                    1333,
                                ),
                            ],
                            type='RandomChoiceResize'),
                    ],
                ],
                type='RandomChoice'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(
        backend_args=None, imdecode_backend='pillow',
        type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        transforms=[
            [
                dict(
                    backend='pillow',
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
            [
                dict(
                    backend='pillow',
                    keep_ratio=True,
                    scales=[
                        (
                            400,
                            1333,
                        ),
                        (
                            500,
                            1333,
                        ),
                        (
                            600,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        384,
                        600,
                    ),
                    crop_type='absolute_range',
                    type='RandomCrop'),
                dict(
                    backend='pillow',
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
        ],
        type='RandomChoice'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='test/_annotations.coco.json',
        backend_args=None,
        data_prefix=dict(img='test/'),
        data_root='./dentex-4',
        metainfo=dict(
            classes=(
                'Teeth',
                'Caries',
                'Deep Caries',
                'Impacted',
                'Periapical Lesion',
            )),
        pipeline=[
            dict(
                backend_args=None,
                imdecode_backend='pillow',
                type='LoadImageFromFile'),
            dict(
                backend='pillow',
                keep_ratio=True,
                scale=(
                    1333,
                    800,
                ),
                type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='./dentex-4/test/_annotations.coco.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(init_kwargs=dict(project='mmdetection'), type='WandbVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './tutorial_exps/dinodisease/swin'
