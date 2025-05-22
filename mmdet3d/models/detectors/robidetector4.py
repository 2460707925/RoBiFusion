import torch
import torch.nn.functional as F 

from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.core import bbox3d2result

from mmdet3d.models import builder
from mmdet3d.models.builder import DETECTORS
from mmdet3d.models.detectors import MVXTwoStageDetector

from mmdet3d.ops import Voxelization

@DETECTORS.register_module()
class BiRobustDetector(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 #img_branch
                 img_backbone=None,
                 img_neck=None,
                 img_bev_encoder=None,
                 #depth_branch    
                 depth_net=None,     
                 loss_depth=None,
                 #pts_branch  
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 #bev_branch
                 bi_fusion_layer=None,
                 fuse_bev_encoder=None,
                 #bev_decoder
                 bev_backbone=None,
                 bev_neck=None, 
                 bev_bbox_head=None,
                 #config  
                 separate_train=False,
                 freeze_img=False,
                 freeze_lidar=False,
                 img_detach=True,
                 # mmdet3d 在build_detector时会添加  
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):
        super(BiRobustDetector,self).__init__(train_cfg=train_cfg,test_cfg=test_cfg,pretrained=pretrained)
        # img_branch
        if img_backbone:
            self.img_backbone=builder.build_backbone(img_backbone)
        else:
            self.img_backbone=None
        
        if img_neck:
            self.img_neck=builder.build_neck(img_neck)
        else:
            self.img_neck=None
            
        if img_bev_encoder:
            self.img_bev_encoder=builder.build_img_bev_encoder(img_bev_encoder)
        else:
            self.img_bev_encoder=None    
        
        # depth_branch
        if depth_net:
            self.depth_net=builder.build_depth_net(depth_net)
        else:
            self.depth_net=None
            
        if loss_depth:
            self.loss_depth=builder.build_loss(loss_depth)
        else:
            self.loss_depth=None
            
        # pts_branch
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        else:
            self.pts_voxel_layer=None         
        
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        else:
            self.pts_voxel_encoder=None
                    
        if pts_middle_encoder:
            self.pts_middle_encoder = builder.build_middle_encoder(pts_middle_encoder)
        else:
            self.pts_middle_encoder=None 
        
        # fuse_branch
        if bi_fusion_layer:
            self.bi_fusion_layer=builder.build_bi_fusion_layer(bi_fusion_layer)
        else:
            self.bi_fusion_layer=None 
        
        if fuse_bev_encoder:
            self.fuse_bev_encoder=builder.build_fusion_bev_encoder(fuse_bev_encoder)
        else:
            self.fuse_bev_encoder=None 
        
        # BEV Decoder
        if bev_backbone:
            self.bev_backbone = builder.build_backbone(bev_backbone)
        else:
            self.bev_backbone=None 
            
        if bev_neck:
            self.bev_neck = builder.build_neck(bev_neck)
        else:
            self.bev_neck=None 
        
        # TransfusionHead
        if bev_bbox_head:
            self.bev_bbox_head=builder.build_head(bev_bbox_head)
        else:
            self.bev_bbox_head=None
        

        # 两条分支独立训练 or 联合训练
        self.separate_train=separate_train
        # 两条分支是否冻结
        self.freeze_img = freeze_img
        self.freeze_lidar = freeze_lidar
        # 在提取点云特征时候是否将图像特征解耦
        self.img_detach=img_detach
        self.log_losses=None
        
        
    def forward_train(self,
                      points=None,
                      img=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_depth=None,
                      ):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_depth_image (list[torch.Tensor],optional): Ground truth depth images.
                Default to None 
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # 获取img_feats,depth_feats,pts_feats
        pts_feats ,depth_feats, context_feats ,pts_coors = self.extract_feats(points, img, img_metas)

        bev_feats,_,_=self.extract_bev_feats(pts_feats,depth_feats,context_feats,pts_coors,img_metas)
        
        losses = dict()
        if gt_depth is not None:
            losses_depth=self.loss_depth(gt_depth,depth_feats)
            losses.update({"losses_depth":losses_depth})
            
        predicts = self.bev_bbox_head(bev_feats)
        # predicts=self.bev_bbox_head(bev_feats,img_metas=img_metas)
        losses_bev = self.bev_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, predicts)
        losses.update(losses_bev)
        self.log_losses=losses
        return losses
    
    def forward_test(self, points=None, img_metas=None, img=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        """Test function without augmentaiton."""
        pts_feats ,depth_feats,context_feats,pts_coors = self.extract_feats(points, img, img_metas)
        bev_feats,_,_=self.extract_bev_feats(pts_feats,depth_feats,context_feats ,pts_coors ,img_metas)
        # outs = self.bev_bbox_head(bev_feats, img_metas=img_metas)
        outs = self.bev_bbox_head(bev_feats)
        bev_bbox_list = self.bev_bbox_head.get_bboxes(outs, img_metas, rescale=False)
        bbox_results = [bbox3d2result(bboxes, scores, labels)
                        for bboxes, scores, labels in bev_bbox_list
                    ]
        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, bev_bbox in zip(bbox_list, bbox_results):
            result_dict['pts_bbox'] = bev_bbox
        return bbox_list

    
    def extract_bev_feats(self,pts_feats,depth_feats,context_feats,pts_coors,img_metas):
        # img_feats为多尺度的特征，目前只有单尺度
        if context_feats is not None and depth_feats is not None:
            img_bev_feats=self.extract_bev_feats_img(context_feats,depth_feats,img_metas)
        else:
            img_bev_feats=None
        
        if pts_feats is not None:
            pts_bev_feats=self.extract_bev_feats_pts(pts_feats,pts_coors)
        else:
            pts_bev_feats=None
        
        fused_bev_feats=self.fuse_bev_encoder(pts_bev_feats,img_bev_feats)        
        fused_bev_feats=self.bev_backbone(fused_bev_feats)
        fused_bev_feats=self.bev_neck(fused_bev_feats)
        
        return fused_bev_feats,pts_bev_feats,img_bev_feats
        
    def extract_bev_feats_pts(self,pts_feats,pts_coors):
        bs=pts_coors[-1,0]+1
        pts_bev_feats=self.pts_middle_encoder(pts_feats,pts_coors,bs)
        return pts_bev_feats
        
    # img_feats[24,256,28,50] depth_feats[24,256,28,50] 
    def extract_bev_feats_img(self,context_feats,depth_feats,img_metas):
        img_bev_feats=self.img_bev_encoder(context_feats,depth_feats,img_metas)
        return img_bev_feats
        
    def extract_feats(self, points, img, img_metas):
        """Extract features from images and points."""
        # 返回的是tuple形式的数据tuple(Tensor(B*N,C,H,W))
        if img is not None:
            img_feats = self.extract_img_feats(img, img_metas) # (24,256,28,50)
            depth_feats,context_feats=self.extract_depth_from_img(img_feats,img_metas)
            # 用于将图像特征与点云特征解耦
            if self.img_detach:
                img_feats = img_feats.detach()
        else:
            img_feats=None
            depth_feats=None
            context_feats=None
            
        if points is not None:
            pts_feats,pts_coors,pts_centroids = self.extract_pts_feats(points)
        else:
            pts_feats=None
            pts_coors=None 
        
        if not self.separate_train:
            pts_feats,depth_feats=self.extract_bi_feats(img_feats,pts_feats,depth_feats,pts_coors,pts_centroids,img_metas)
        return pts_feats,depth_feats,context_feats,pts_coors


    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feats(self, img, img_metas):
        """Extract features of images."""
        img_feats=None
        if self.with_img_backbone:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)

            img_feats = self.img_backbone(img.float())
            
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        
        return img_feats
    
    def extract_depth_from_img(self,img_feats,img_metas):
        depth_feats,context_feats=self.depth_net(img_feats, img_metas)
        return depth_feats,context_feats
    
    def extract_pts_feats(self,points):
        voxels,coors=self.voxelize(points)
        
        voxel_feats,voxel_coors,voxel_centroids=self.pts_voxel_encoder(voxels,coors)
        
        return voxel_feats,voxel_coors,voxel_centroids

    def extract_bi_feats(self,img_feats,pts_feats,depth_feats,pts_coors,pts_centroids,img_metas):
        pts_feats,depth_feats=self.bi_fusion_layer(img_feats,pts_feats,depth_feats,pts_coors,pts_centroids,img_metas)
        return pts_feats,depth_feats
    
    

    # 返回带语义信息的点云特征和
    def extract_bi_pts_feats(self, points, img_feats, img_metas):
        """Extract point features."""
        voxels, coors = self.voxelize(points)

        voxel_features, voxel_coors, depth_features = self.pts_voxel_encoder(
            voxels, coors, img_feats, img_metas)  
            
        return voxel_features,depth_features
    

    @torch.no_grad()
    @force_fp32(apply_to=('points'))
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            # (x,y,z)->(z,y,x)
            # res_coors = self.pts_voxel_layer(res).flip(dims=[-1])
            res_coors = self.pts_voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch
 