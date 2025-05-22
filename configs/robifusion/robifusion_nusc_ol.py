dataset_type = 'NuScenesDataset'
data_root = './data/nuscenes/'
gt_paste_stop_epoch = 15
reduce_beams = 32
load_dim = 5
use_dim = 5
load_augmented = None
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.15, 0.15, 0.2]
out_size_factor = 4
grid_size=[int((point_cloud_range[3]-point_cloud_range[0])/voxel_size[0]),
           int((point_cloud_range[4]-point_cloud_range[1])/voxel_size[1]),
           int((point_cloud_range[5]-point_cloud_range[2])/voxel_size[2])]
augment2d = dict(
    resize=[[0.38, 0.55], [0.48, 0.48]],
    rotate=[-5.4, 5.4],
    gridmask=dict(prob=0.0, fixed_prob=True))
augment3d = dict(
    rot_range=[0, 0], scale_ratio_range=[1.0, 1.0], translation_std=[0, 0, 0])
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4]),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='ObjectSample',
        db_sampler=dict(
            data_root='./data/nuscenes/',
            info_path=
            './data/nuscenes/nuscenes_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(
                    car=5,
                    truck=5,
                    bus=5,
                    trailer=5,
                    construction_vehicle=5,
                    traffic_cone=5,
                    barrier=5,
                    motorcycle=5,
                    bicycle=5,
                    pedestrian=5)),
            classes=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            sample_groups=dict(
                car=2,
                truck=3,
                construction_vehicle=7,
                bus=4,
                trailer=6,
                barrier=2,
                motorcycle=6,
                bicycle=6,
                pedestrian=2,
                traffic_cone=2),
            points_loader=dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=[0, 1, 2, 3, 4]))),
    dict(
        type='GlobalRotScaleTrans',
        rot_lim=[0, 0],
        resize_lim=[1.0, 1.0],
        trans_lim=[0, 0, 0],
        is_train=False),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ]),
    dict(type='PointShuffle'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ]),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4]),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4]),
    dict(
        type='GlobalRotScaleTrans',
        rot_lim=[0, 0],
        resize_lim=[1.0, 1.0],
        trans_lim=[0, 0, 0],
        is_train=False),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type='NuScenesDataset',
            data_root='./data/nuscenes/',
            ann_file=
            './data/nuscenes/nuscenes_infos_train.pkl',
            load_interval=5,
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=[0, 1, 2, 3, 4]),
                dict(
                    type='LoadPointsFromMultiSweeps',
                    sweeps_num=10,
                    use_dim=[0, 1, 2, 3, 4]),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='ObjectSample',
                    db_sampler=dict(
                        data_root='./data/nuscenes/',
                        info_path=
                        './data/nuscenes/nuscenes_dbinfos_train.pkl',
                        rate=1.0,
                        prepare=dict(
                            filter_by_difficulty=[-1],
                            filter_by_min_points=dict(
                                car=5,
                                truck=5,
                                bus=5,
                                trailer=5,
                                construction_vehicle=5,
                                traffic_cone=5,
                                barrier=5,
                                motorcycle=5,
                                bicycle=5,
                                pedestrian=5)),
                        classes=[
                            'car', 'truck', 'trailer', 'bus',
                            'construction_vehicle', 'bicycle', 'motorcycle',
                            'pedestrian', 'traffic_cone', 'barrier'
                        ],
                        sample_groups=dict(
                            car=2,
                            truck=3,
                            construction_vehicle=7,
                            bus=4,
                            trailer=6,
                            barrier=2,
                            motorcycle=6,
                            bicycle=6,
                            pedestrian=2,
                            traffic_cone=2),
                        points_loader=dict(
                            type='LoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=5,
                            use_dim=[0, 1, 2, 3, 4]))),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_lim=[0, 0],
                    resize_lim=[1.0, 1.0],
                    trans_lim=[0, 0, 0],
                    is_train=False),
                dict(
                    type='PointsRangeFilter',
                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(
                    type='ObjectNameFilter',
                    classes=[
                        'car', 'truck', 'trailer', 'bus',
                        'construction_vehicle', 'bicycle', 'motorcycle',
                        'pedestrian', 'traffic_cone', 'barrier'
                    ]),
                dict(type='PointShuffle'),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'trailer', 'bus',
                        'construction_vehicle', 'bicycle', 'motorcycle',
                        'pedestrian', 'traffic_cone', 'barrier'
                    ]),
                dict(
                    type='Collect3D',
                    keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
            ],
            classes=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            modality=dict(
                use_lidar=True,
                use_camera=False,
                use_radar=False,
                use_map=False,
                use_external=False),
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type='NuScenesDataset',
        data_root='./data/nuscenes/',
        ann_file=
        './data/nuscenes/nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=[0, 1, 2, 3, 4]),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                use_dim=[0, 1, 2, 3, 4]),
            dict(
                type='GlobalRotScaleTrans',
                rot_lim=[0, 0],
                resize_lim=[1.0, 1.0],
                trans_lim=[0, 0, 0],
                is_train=False),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ],
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type='NuScenesDataset',
        data_root='./data/nuscenes/',
        ann_file=
        './data/nuscenes/nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=[0, 1, 2, 3, 4]),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                use_dim=[0, 1, 2, 3, 4]),
            dict(
                type='GlobalRotScaleTrans',
                rot_lim=[0, 0],
                resize_lim=[1.0, 1.0],
                trans_lim=[0, 0, 0],
                is_train=False),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ],
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR'))
model = dict(
    type='BiRobustDetector',
    separate_train=True,
    pts_voxel_layer=dict(
        point_cloud_range=point_cloud_range,
        max_num_points=-1,
        voxel_size=voxel_size,
        max_voxels=(-1, -1)),
    pts_voxel_encoder=dict(
        type='MyDynamicVFE',
        in_channels=5,
        feat_channels=[64, 64],
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        mode='max',
        return_point_feats=False,
        return_centroids=True),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=64,
        sparse_shape=[grid_size[2], grid_size[0], grid_size[1]],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock',
        up_cfg=dict(
            upsample_cfg=dict(type='deconv', bias=False),
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        )
        ),
        
    fuse_bev_encoder=dict(
        type='BEVConvFuser',
        in_channels=[128],
        out_channels=256,
    ),

    # BEV Decoder
    bev_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    
    bev_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256,256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    
    bev_bbox_head=dict(
        type='CenterHead',
        in_channels=512,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        train_cfg=dict(
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=out_size_factor,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]),
        test_cfg=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            pc_range=point_cloud_range,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size,
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)
    ),
)
custom_hooks = [dict(type='PasteStopHook', stop_epoch=15)]
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)  # for 8gpu * 2sample_per_gpu
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
total_epochs = 20
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 4)
