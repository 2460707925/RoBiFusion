import math
import numpy as np 

import torch
import torch.nn as nn
from torch.nn import functional as F

from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule, xavier_init, normal_init, constant_init
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch
from mmdet3d.core.bbox.structures import points_cam2img
from mmdet3d.models.fusion_layers.coord_transform import apply_3d_transformation
from mmdet3d.ops.round_grid_sample import RoundGridSample

from .points_depth import GenerateDepthFeature_Pts

class Atten_Layer(nn.Module):
    def __init__(self, channels):
        super(Atten_Layer, self).__init__()
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)


    def forward(self, img_feats, point_feats):
        ri = self.fc1(img_feats)
        rp = self.fc2(point_feats)
        att = torch.sigmoid(self.fc3(torch.tanh(ri + rp))) 

        return att*img_feats

class PointAlignModule(BaseModule):
    def __init__(self,
                 img_channels,
                 pts_channels,
                 mid_channels,
                 out_channels,
                 depth_channels,
                 coord_type='LIDAR',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None,
                 activate_out=True,
                 dropout_ratio=0,
                 aligned=True,
                 align_corners=True,
                 padding_mode='zeros',
                 num_heads=4,
                 num_points=8,
                 fix_offset=True,
                 im2col_step=64,
                 multi_input='',
                 intersect_type='maxpool',
                 lateral_conv=True):
        super(PointAlignModule, self).__init__(init_cfg=init_cfg)
        
        self.img_channels=img_channels
        self.pts_channels=pts_channels
        self.mid_channels=mid_channels
        self.out_channels=out_channels
        self.depth_channels=depth_channels
        
        self.coord_type = coord_type
        self.act_cfg = act_cfg
        self.activate_out = activate_out
        self.dropout_ratio = dropout_ratio
        self.aligned = aligned
        self.align_corners = align_corners
        self.padding_mode = padding_mode
        self.num_points = num_points
        self.num_heads = num_heads
        self.fix_offset = fix_offset
        self.im2col_step = im2col_step
        self.multi_input = multi_input
        self.generate_depth=GenerateDepthFeature_Pts(depth_dim=depth_channels)
        
        # 用于多尺度特征的，目前只支持单尺度
        if lateral_conv:
            self.lateral_conv = ConvModule(
                self.img_channels,
                self.mid_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
        else:
            assert self.img_channels==self.mid_channels,"img_channels!=mid_channels"
            self.lateral_conv=None
        
        self.img_proj=nn.Linear(self.mid_channels,self.mid_channels)
        self.pts_proj=nn.Linear(self.pts_channels,self.mid_channels)
        self.value_proj = nn.Linear(self.mid_channels, self.out_channels)
        
        self.deform_sampling_offsets = nn.Linear(self.mid_channels, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(self.mid_channels, num_heads * num_points)
        
        self.depth_confidence_mlp = nn.Linear(self.mid_channels, 1)

        # 特征融合模块
        self.atten_layer=Atten_Layer([self.out_channels,self.pts_channels])
        self.fusion_layer= nn.Linear(self.pts_channels + self.out_channels, self.pts_channels)
        self.bn1 = nn.BatchNorm1d(self.pts_channels)

        if self.fix_offset:
            self.deform_sampling_offsets.weight.requires_grad = False
            self.deform_sampling_offsets.bias.requires_grad = False

        if intersect_type=='maxpool':
            self.intersect_layer=nn.MaxPool1d(kernel_size=2)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.Conv2d):
                normal_init(m, 0., std=1.0)
        constant_init(self.attention_weights, val=0., bias=0.)
        constant_init(self.deform_sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.deform_sampling_offsets.bias.data = grid_init.view(-1)

    def forward(self,
                img_feats,
                voxel_feats,
                voxel_coors,
                voxel_centroids,
                img_metas,
                voxel_size,
                point_cloud_range):
        """
        img_feats(list(B,num_camera,C,H1,W1)
        voxel_coors(N,4)
        voxel_centroids(N,3)
        voxel_feats(N,64)
        img_feats([(B,num_camera,C,H,W)])
        """
        # 在deformable attention的时候使用了voxel_feats
        img_pts, point2depth = self.obtain_feats(img_feats, voxel_coors, voxel_centroids ,voxel_feats,
                                                    img_metas, voxel_size, point_cloud_range)
        fuse_out=self.li_fusion_layer(voxel_feats,img_pts)
        return fuse_out,point2depth
    
    def li_fusion_layer(self,lidar_feat,image_feat):
        if self.training and self.dropout_ratio > 0:
            image_feat = F.dropout(image_feat, self.dropout_ratio)
        image_feat=self.atten_layer(image_feat,lidar_feat)     
        fuse_feat=torch.concat([lidar_feat,image_feat],dim=-1)
        fuse_feat=F.relu(self.bn1(self.fusion_layer(fuse_feat)))
        return fuse_feat


    # 获取mlvl_feats的特征(可以进行优化)
    """
    img_feats(list[(B,num_camera,C,H1,W1),...])
    voxel_coors(N,4)
    voxel_feats(N,64)
    img_feats([(B,num_camera,C,H,W)]*num_level)
    """
    def obtain_feats(self,
                     img_feats,
                     voxel_coors,
                     voxel_centroids,
                     voxel_feats,
                     img_metas,
                     voxel_size,
                     point_cloud_range):
        """Obtain multi-level features for each point.

        Args:
            img_feats (list(torch.Tensor)): Multi-scale image features produced
                by image backbone in shape (N, C, H, W).
            pts (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Meta information for each sample.

        Returns:
            torch.Tensor: Corresponding image features of each point.
        """
        batch_size=len(img_metas)
        num_camera=img_feats.shape[0]//batch_size
        _,_,img_height,img_width=img_feats.shape
        
        if self.lateral_conv is not None:
            img_ins = self.lateral_conv(img_feats)
        else:
            img_ins = img_feats

        img_ins=img_ins.view(batch_size,num_camera,-1,img_height,img_width)
        
        start_iter = 0
        sample_img_feats = []
        sample_depth_feature_maps=[]
        for i in range(batch_size):
            voxel_coors_per_img = voxel_coors[voxel_coors[:, 0] == i]
            x = (voxel_coors_per_img[:, 3] + 0.5) * \
                voxel_size[0] + point_cloud_range[0]
            y = (voxel_coors_per_img[:, 2] + 0.5) * \
                voxel_size[1] + point_cloud_range[1]
            z = (voxel_coors_per_img[:, 1] + 0.5) * \
                voxel_size[2] + point_cloud_range[2]
            x = x.unsqueeze(-1)
            y = y.unsqueeze(-1)
            z = z.unsqueeze(-1)

            voxel_coors_per_batch = torch.cat([x, y, z], dim=-1)
            num_voxels = voxel_coors_per_batch.shape[0]
            voxel_centroids_per_batch=voxel_centroids[start_iter:start_iter+num_voxels]
            voxel_feats_per_batch = voxel_feats[start_iter: start_iter + num_voxels]
            # 对每一个batch进行img_feat进行提取
            sample_img_feats_per_batch, sample_depth_feature_maps_per_batch = self.sample_single(img_ins[i],
                                                                                        voxel_centroids_per_batch,
                                                                                        voxel_feats_per_batch,
                                                                                        img_metas[i])
            sample_img_feats.append(sample_img_feats_per_batch)
            sample_depth_feature_maps.append(sample_depth_feature_maps_per_batch)
            
            start_iter += num_voxels
        # (B*N,C)
        img_pts = torch.cat(sample_img_feats, dim=0)
        # (B*N,H,W,C)->(B*N,C,H,W)
        sample_depth_feature_maps=torch.concat(sample_depth_feature_maps,dim=0).permute(0,3,1,2)
        
        return img_pts,sample_depth_feature_maps


    def sample_single(self, img_feats, voxel_centroids ,voxel_feats, img_meta):
        """Sample features from single level image feature map.
        Args:
            img_feats (list(torch.Tensor)): Image feature map in shape 
                (num_camera, C, H, W). len(img_feats)=img_levels
            pts (torch.Tensor): Points of a single sample.
            img_meta (dict): Meta information of the single sample.

        Returns:
            torch.Tensor: Single level image features of each point.
        """
        # TODO: image transformation also extracted
        num_camera,_,img_height,img_width = img_feats.shape
        feature_shape=(img_height,img_width)
        
        lidar2image_matrix = img_meta["lidar2image"].data
        img_aug_matrix = img_meta["img_aug_matrix"].data
        # align pts_feats from each camera
        assign_mask = voxel_centroids.new_zeros((voxel_centroids.shape[0]), dtype=torch.bool)
        final_img_pts = voxel_centroids.new_zeros((voxel_centroids.shape[0], self.mid_channels))

        multi_depth_feature_maps=[]
        
        from time import time 
        times=[]
        for camera_id in range(num_camera):
            t1_0=time()
            grid, valid_idx, grid_depth = point_sample(
                img_meta=img_meta,
                points=voxel_centroids,
                proj_mat=lidar2image_matrix[camera_id].to(voxel_centroids.device),
                img_aug_mat=img_aug_matrix[camera_id].to(voxel_centroids.device),
                coord_type=self.coord_type,
                img_shape=img_meta['input_shape'][:2],
            )
            
            if torch.sum(valid_idx!=0)==0:
                multi_depth_feature_maps.append(torch.zeros((img_height,img_width,self.depth_channels),device=img_feats.device))
                continue
            
            assign_idx = (~assign_mask) & valid_idx
            intersect_idx = assign_mask & valid_idx
            assign_mask |= assign_idx
            
            # 后面这里应该是torch.gather的操作会比较好
            assign_grid = grid[assign_idx].unsqueeze(0).unsqueeze(0)

            intersect_grid = grid[intersect_idx].unsqueeze(0).unsqueeze(0)
            
            assign_num = assign_grid.shape[2]
            
            assign_voxel_feats = voxel_feats[assign_idx].unsqueeze(0)
            intersect_voxel_feats = voxel_feats[intersect_idx].unsqueeze(0)
            
            if camera_id==0:
                valid_grid=assign_grid
                valid_voxel_feats=assign_voxel_feats
                valid_grid_depth=grid_depth[assign_idx]
            else:
                valid_grid=torch.cat([assign_grid,intersect_grid],dim=2)
                valid_voxel_feats = torch.cat([assign_voxel_feats,intersect_voxel_feats],dim=1)
                valid_grid_depth=torch.cat([grid_depth[assign_idx],grid_depth[intersect_idx]],dim=0)
            
            # align_corner=True provides higher performance
            mode="bilinear"
            if mode in ["bilinear", "nearest"]:
                valid_ref_feats = F.grid_sample(
                    # img_feats:(num_camera,C,H,W) result:(1,C,H,W)
                    img_feats[camera_id].unsqueeze(0),
                    valid_grid,  # valid_grid:()
                    mode=mode,
                    padding_mode='zeros',
                    align_corners=False).squeeze(-2).permute(0, 2, 1)
                # 1xCx1xN
                # 1xCx1xN->1xCxN->1xNxC
            else:
                valid_ref_feats = RoundGridSample.apply(
                    img_feats[camera_id].unsqueeze(0),
                    valid_grid.clone(),
                ).permute(0, 2, 1)


            valid_ref_feats = self.img_proj(valid_ref_feats)
            valid_voxel_feats = self.pts_proj(valid_voxel_feats.detach())
            # valid_voxel_feats = self.pts_proj(valid_voxel_feats)

            # use the multiply_pts_detach
            if self.multi_input == 'concat':
                query_feat = torch.cat(
                    [valid_ref_feats, valid_voxel_feats], dim=-1)
            elif self.multi_input == 'multiply':
                query_feat = valid_ref_feats * valid_voxel_feats
            # elif self.multi_input == 'multiply_pts_detach':
            #     query_feat = valid_ref_feats * valid_voxel_feats.detach()
            elif self.multi_input == 'pts':
                query_feat = valid_voxel_feats
            elif self.multi_input == 'pts_detach':
                query_feat = valid_voxel_feats.detach()
            elif self.multi_input == 'img':
                query_feat = valid_ref_feats
            elif self.multi_input == 'img_detach':
                query_feat = valid_ref_feats.detach()
            else:
                raise Exception
            num_query = query_feat.shape[1]

            sampling_offsets = self.deform_sampling_offsets(query_feat).view(
                1, num_query, self.num_heads, self.num_points, 2
            )
            attention_weights = self.attention_weights(query_feat).view(
                1, num_query, self.num_heads, self.num_points
            )
            attention_weights = attention_weights.softmax(-1).unsqueeze(3)

            valid_grid = valid_grid.permute(0, 2, 1, 3)

            spatial_shapes = []

            img_feats_per_camera = img_feats[camera_id]

            _, h, w = img_feats_per_camera.shape
            spatial_shapes.append((h, w))
            flatten_feat = img_feats_per_camera.flatten(1).transpose(0,1)
            value_flatten = self.value_proj(flatten_feat)         
            num_value, _ = value_flatten.shape
            value_flatten = value_flatten.view(1, num_value, self.num_heads, -1)
            
            spatial_shapes = torch.as_tensor(
                spatial_shapes, dtype=torch.long, device=flatten_feat.device)
            
            level_start_index = torch.cat((spatial_shapes.new_zeros(
                (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))


            assert valid_grid.shape[-1] == 2, f"Last dim of reference_points must be \
                2 or 4, but get {valid_grid.shape[-1]} instead."
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = valid_grid[:, :, None, :, None, :] \
                + sampling_offsets[:,:,:,None,:,:] / offset_normalizer[None, None, None, :, None, :]

            # if torch.cuda.is_available() and value_flatten.is_cuda:
            #     # value_flatten.shape=[1,15120,4,16]
            #     # spatial_shapes=[[80,144],[40,72],[20,36]]
            #     # level_start_index=[0,11520,14400]
            #     # sampling_locations.shape=[1,6836,4,3,8,2]
            #     # attention_weights.shape=[1,6836,4,24]
            #     # self.im2col_step=64
            #     output = MultiScaleDeformableAttnFunction.apply(
            #         value_flatten, spatial_shapes, level_start_index, sampling_locations,
            #         attention_weights, self.im2col_step)
            # else:
            #     # WON'T REACH HERE
            #     print("Won't Reach Here")
            output = multi_scale_deformable_attn_pytorch(
                value_flatten, spatial_shapes, sampling_locations, attention_weights)
            
            # output.shape=[1,6836,64]
            output = (output + valid_ref_feats).squeeze(0)
            
            # final_img_pts.shape=[26121,64]
            if camera_id==0:
                final_img_pts[assign_idx]=output[:assign_num]
            else:
                final_img_pts[assign_idx] = output[:assign_num]
                
                final_img_pts[intersect_idx] = self.intersect_layer(torch.stack((final_img_pts[intersect_idx],output[assign_num:]),dim=2)).squeeze(-1)
           
            # depth prediction
            # 在这里可以预测四个点，后面可以说是对于点的偏移量的修正
            t1_1=time()
            confs=torch.tanh(self.depth_confidence_mlp(query_feat).view(num_query, 1)).exp()
            t1_2=time()
            valid_depth_pos=valid_grid.reshape(-1,2).detach()
            valid_grid_depth=valid_grid_depth.detach().unsqueeze(-1).int()
            multi_depth_feature_maps.append(self.generate_depth(valid_depth_pos,valid_grid_depth,confs,feature_shape)) 
            t1_3=time()
            total_time=t1_3-t1_0
            times.append((t1_1-t1_0,t1_2-t1_1,t1_3-t1_2,total_time))
        # t1,t2,t3,tt=0,0,0,0
        # for time in times:
        #     t1+=time[0]
        #     t2+=time[1]
        #     t3+=time[2]
        #     tt+=time[3]
        # print(f"time cal:{t1},{t2},{t3},{tt}")
        # print(t1/tt,t2/tt,t3/tt) 
   
        multi_depth_feature_maps=torch.stack(multi_depth_feature_maps,dim=0)

        return final_img_pts,multi_depth_feature_maps
        
    
def point_sample(img_meta,
                 points,
                 proj_mat,
                 img_aug_mat,
                 coord_type,
                 img_shape):
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta info.
        img_features (torch.Tensor): 1 x C x H x W image features.
        points (torch.Tensor): Nx3 point cloud in LiDAR coordinates.
        proj_mat (torch.Tensor): 4x4 transformation matrix.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (torch.Tensor): Scale factor with shape of \
            (w_scale, h_scale).
        img_crop_offset (torch.Tensor): Crop offset used to crop \
            image during data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_shape (tuple[int]): int tuple indicates the h & w after
            padding, this is necessary to obtain features in feature map.
        img_shape (tuple[int]): int tuple indicates the h & w before padding
            after scaling, this is necessary for flipping coordinates.
        aligned (bool, optional): Whether use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str, optional): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool, optional): Whether to align corners when
            sampling image features for each point. Defaults to True.

    Returns:
        torch.Tensor: NxC image features sampled by point coordinates.
    """

    # apply transformation based on info in img_meta
    # 将points转换为原来的点云x,y,z
    points = apply_3d_transformation(
        points[:,:3], coord_type, img_meta, reverse=True)

    # project points to camera coordinate
    pts_2d_with_depth = points_cam2img(points, proj_mat.to(points.device), with_depth=True)
    pts_depth = pts_2d_with_depth[:, -1]
    pts_2d = pts_2d_with_depth[:, :2]

    valid_depth_idx = (pts_depth > 0).reshape(-1,)

    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[:, :2] @ img_aug_mat[:2,:2].T+img_aug_mat[:2,3].T

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

    # if img_flip:
    #     # by default we take it as horizontal flip
    #     # use img_shape before padding for flip
    #     orig_h, orig_w = img_shape
    #     coor_x = orig_w - coor_x

    h, w = img_shape
    coor_y = coor_y / h * 2 - 1
    coor_x = coor_x / w * 2 - 1

    valid_y_idx = ((coor_y >= -1) & (coor_y <= 1)).reshape(-1,)
    valid_x_idx = ((coor_x >= -1) & (coor_x <= 1)).reshape(-1,)

    grid = torch.cat([coor_x, coor_y],
                     dim=1)

    valid_idx = valid_depth_idx & valid_y_idx & valid_x_idx
    return grid, valid_idx, pts_depth
            

