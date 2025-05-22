# compared with robifusion_lc.py without bifusion module 
dataset_type = 'NuScenesDataset'
data_root = '/home/jovyan/data/nuscenes/nuscenes_mini/'

reduce_beams=32
load_dim=5
use_dim=[0,1,2,3,4]

point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size=[0.1,0.1,0.2]
dbound=[1.0,51.0,0.5]
depth_channels=int((dbound[1]-dbound[0])//dbound[2])
img_size=[448,800]
out_size_factor=8
grid_size=[int((point_cloud_range[3]-point_cloud_range[0])/voxel_size[0]),
           int((point_cloud_range[4]-point_cloud_range[1])/voxel_size[1]),
           int((point_cloud_range[5]-point_cloud_range[2])/voxel_size[2])]
augment2d=dict(resize=[[0.5, 0.6], [0.48, 0.48]],
               rotate=[-5.4, 5.4],
               gridmask=dict(prob=0.0,fixed_prob=True)
               )

augment3d=dict(
    rot_range=[0, 0],
    scale_ratio_range=[1.0, 1.0],
    translation_std=[0, 0, 0]
)

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

num_views = 6
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline=[
    # Lidar分支加载
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim,
        # reduce_beams=reduce_beams,
    ),
    
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=use_dim,
        # reduce_beams=reduce_beams,
    ),
    
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True,
         with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_lim=augment3d["rot_range"],
        resize_lim=augment3d["scale_ratio_range"],
        trans_lim=augment3d["translation_std"],
        is_train=False,
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    

    # Camera分支加载
    dict(type='LoadMultiViewImageFromFiles',to_float32=True),
    
    dict(type="ImageAug3D",
         final_dim=img_size,
         resize_lim=augment2d["resize"][0],
         bot_pct_lim= [0.0, 0.0],
         rot_lim= augment2d["rotate"],
         rand_flip= False,
         is_train= True,
         ),

    dict(type="ImageNormalize",
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225],
    ),
    dict(type='DefaultFormatBundle3D',  class_names=class_names),
    dict(type='GTDepth',keyframe_only=False),
    
    dict(type='Collect3D',
         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d','gt_depth'],
         meta_keys=['camera_intrinsics','camera2ego','lidar2ego','lidar2camera',
                    'camera2lidar','lidar2image','img_aug_matrix','lidar_aug_matrix',
                    'img_shape','pad_shape','scale_factor','img_norm_cfg'],
        )
]


test_pipeline = [
    # Lidar分支加载
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim,
        # reduce_beams=reduce_beams,
    ),
    
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=use_dim,
        # reduce_beams=reduce_beams,
    ),
    
    dict(
        type='GlobalRotScaleTrans',
        rot_lim=augment3d["rot_range"],
        resize_lim=augment3d["scale_ratio_range"],
        trans_lim=augment3d["translation_std"],
        is_train=False,
    ),
    

    # Camera分支加载
    dict(type='LoadMultiViewImageFromFiles',to_float32=True),
    
    dict(type="ImageAug3D",
         final_dim=img_size,
         resize_lim=augment2d["resize"][0],
         bot_pct_lim= [0.0, 0.0],
         rot_lim= augment2d["rotate"],
         rand_flip= False,
         is_train= False,
         ),

    dict(type="ImageNormalize",
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225],
    ),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    
    dict(type='Collect3D',
         keys=['points', 'img'],
         meta_keys=['camera_intrinsics','camera2ego','lidar2ego','lidar2camera',
                    'camera2lidar','lidar2image','img_aug_matrix','lidar_aug_matrix',
                    'img_shape','pad_shape','scale_factor','img_norm_cfg','box_type_3d'],
        )
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + '/nuscenes_infos_train.pkl',
            load_interval=1,
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))


model = dict(
    type='BiRobustDetector',
    separate_train=True,
    # Encoder(Camera)
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint="checkpoints/resnet50.pth"),
        ),
    
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256,512,1024,2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[64, 64, 64, 64],
        ),
    
    img_bev_encoder=dict(
        type='ImgBEVGeneration',
        x_bound=[-51.2, 51.2, 0.8],
        y_bound=[-51.2, 51.2, 0.8],
        z_bound=[-5, 3, 8],
        d_bound=dbound,
        final_dim=img_size,
        output_channels=80,
        downsample_factor=16,
        ),
    
    depth_net=dict(
        type='DepthInitialModule',
        in_channels=256,
        mid_channels=256,
        context_channels=80,
        depth_channels=depth_channels,
    ),
    
    loss_depth=dict(
        type='DepthLoss',
        downsample_factor=16,
        dbound=dbound,
        depth_loss_factor=3.0,
    ),
    
    # Lidar分支
    pts_voxel_layer=dict(
        point_cloud_range=point_cloud_range,
        max_num_points=-1, voxel_size=voxel_size, max_voxels=(-1, -1)),
    
    pts_voxel_encoder=dict(
        type='MyDynamicVFE',
        in_channels=5,
        feat_channels=[64,64],
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        mode='max',
        return_point_feats=False,
        return_centroids=True,
    ),
    
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=64,
        sparse_shape=[41, 1024, 1024],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    
    # Fusion融合
    bi_fusion_layer=dict(
        type="BiFusionModule",
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        point_align_config={"img_channels":256,
                            "pts_channels":64,
                            "mid_channels":64,
                            "out_channels":64,
                            "depth_channels":depth_channels,
                            "multi_input":'multiply_pts_detach'},
        depth_refinement_config={"img_depth_norm":'p1',
                                 "pts_depth_norm":'p1',
                                 "merge_depth_norm":'p2'}
    ),
    
    # 应该在这里加入drop训练，丢失一个模态进行训练
    fuse_bev_encoder=dict(
        type='BEVConvFuser',
        in_channels=[80,256],
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
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 8)