import numpy as np 
import torch 
import torch.nn as nn 
from torch.cuda.amp.autocast_mode import autocast

from ..builder import IMG_BEV_ENCODER

from mmdet3d.ops.voxel_pooling_train import voxel_pooling_train
from mmdet3d.ops.voxel_pooling_inference import voxel_pooling_inference

            
@IMG_BEV_ENCODER.register_module()            
class ImgBEVGeneration(nn.Module):

    def __init__(self,
                 x_bound,
                 y_bound,
                 z_bound,
                 d_bound,
                 final_dim,
                 output_channels,
                 downsample_factor,
                 use_da=True):
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.

        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            depth_net_conf (dict): Config for depth net.
        """

        super(ImgBEVGeneration, self).__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels=output_channels

        self.register_buffer(
            'voxel_size',
            torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            'voxel_coord',
            torch.Tensor([
                row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]
            ]))
        self.register_buffer(
            'voxel_num',
            torch.LongTensor([(row[1] - row[0]) / row[2]
                              for row in [x_bound, y_bound, z_bound]]))
        
        # D,H,W,3
        self.register_buffer('frustum', self.create_frustum())
        self.depth_channels, _, _, _ = self.frustum.shape
        self.use_da=use_da
        if self.use_da:
            self.depth_aggregation_net=DepthAggregation(self.output_channels, self.output_channels,self.output_channels)
        
        
    # 制造视锥，比如在真实长宽深度(ogfH:256,ogfW:704,depth:2~58)中生成(16,44,112)的voxel,最后返回对应的真实坐标
    # 真实坐标(x_coords,y_coords,d_coords,paddings
    def create_frustum(self):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        # d_coords:(D,H,W)
        d_coords = torch.arange(*self.d_bound,
                                dtype=torch.float).view(-1, 1,
                                                        1).expand(-1, fH, fW)
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum
            
            
    def get_geometry(self, img_metas):
        """Transfer points from camera coord to ego coord.

        Args:
            sensor2ego_mats(Tensor): Transformation matrix from
                camera to ego with shape of (B, num_cameras, 4, 4).                
            intrin_mats(Tensor): Intrinsic matrix with shape
                of (B, num_cameras, 4, 4).
            ida_mats(Tensor): Transformation matrix for ida with
                shape of (B, num_sweeps, num_cameras, 4, 4).
            bda_mat(Tensor): Rotation matrix for bda with shape
                of (B, 4, 4).

        Returns:
            Tensors: points ego coord.
        """
        points = self.frustum
        sensor2ego_mat,intrin_mat,ida_mat,bda_mat=self.get_image_parameters(img_metas,points.device)
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        # if not points.is_cuda:
        #     points=points.cuda()
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        # batch_size,num_cams,depth_dim,h_dim,w_dim,4,1
        
        points = ida_mat.inverse().matmul(points.unsqueeze(-1))
        # cam_to_ego
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:]), 5)

        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4,
                              4).matmul(points)
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        # 输出（B，M, D, H, W, 3）->（B*M*D*H*W，3）
        # 比如你知道在b1,第m1个摄像头上，他的(d1,h1,w1)网格所对应的真实位置的坐标,做了一个预先的投影设置
        return points[..., :3]
    
    
    def forward(self,
                img_feats,
                depth_feats,
                img_metas):
        batch_size=len(img_metas)
        num_cams=img_feats.shape[0]//batch_size
        # 视锥生成,因为在同一个像素坐标下,不同的深度对应的x,y,z并不是对应的一个线性增长,
        # 所以我们提前计算好不同的depth对应的BEV特征
        geom_xyz=self.get_geometry(img_metas)
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()

        if self.training or self.use_da:
            img_feat_with_depth = depth_feats.unsqueeze(1) * img_feats.unsqueeze(2)

            img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)

            img_feat_with_depth = img_feat_with_depth.reshape(
                batch_size,
                num_cams,
                img_feat_with_depth.shape[1],
                img_feat_with_depth.shape[2],
                img_feat_with_depth.shape[3],
                img_feat_with_depth.shape[4],
            )

            # img_feat_with_depth.shape=(1,6,256,16,44,256)
            img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)

            feature_map = voxel_pooling_train(geom_xyz,
                                              img_feat_with_depth.contiguous(),
                                              self.voxel_num.cuda())
        else:
            feature_map = voxel_pooling_inference(geom_xyz, 
                                                  depth_feats, 
                                                  img_feats, 
                                                  self.voxel_num.cuda())
        return feature_map.contiguous()
        
    def get_image_parameters(self,img_metas,device):
        batch_size=len(img_metas)
        sensor2ego_mats=[]
        intrin_mats=[]
        ida_mats=[]
        bda_mats=[]
        for i in range(batch_size):
            img_meta=img_metas[i]
            sensor2ego_mats.append(img_meta['camera2lidar'].data)
            intrin_mats.append(img_meta['camera_intrinsics'].data)
            ida_mats.append(img_meta['img_aug_matrix'].data)
            bda_mats.append(img_meta['lidar_aug_matrix'].data)
        sensor2ego_mats=torch.stack(sensor2ego_mats,dim=0).to(device=device)
        intrin_mats=torch.stack(intrin_mats,dim=0).to(device=device)
        ida_mats=torch.stack(ida_mats,dim=0).to(device=device)
        bda_mats=torch.stack(bda_mats,dim=0).to(device=device)
        return sensor2ego_mats,intrin_mats,ida_mats,bda_mats
        
    # img_feat_with_depth (24,256,256,28,50)
    def _forward_voxel_net(self, img_feat_with_depth):
        if self.use_da:
            # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
            img_feat_with_depth = img_feat_with_depth.permute(0, 3, 1, 4, 2).contiguous()  
            # [n, c, d, h, w] -> [n, h, c, w, d]
            n, h, c, w, d = img_feat_with_depth.shape
            img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d).contiguous()
            
            img_feat_with_depth = (self.depth_aggregation_net(img_feat_with_depth).view(n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous())
        # [n ,c ,d, h, w]
        return img_feat_with_depth
    
 
class DepthAggregation(nn.Module):
    """
    pixel cloud feature extraction
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                        mid_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                        mid_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                        mid_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x       