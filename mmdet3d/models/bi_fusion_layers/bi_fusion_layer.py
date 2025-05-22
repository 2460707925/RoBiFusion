import torch.nn as nn

from ..builder import BI_FUSION_LAYERS
from .point_align import PointAlignModule,RoundGridSample
from .depth_refinement import DepthRefinementModule

@BI_FUSION_LAYERS.register_module()
class BiFusionModule(nn.Module):
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 point_align_config={"img_channels":256,
                                     "pts_channels":64,
                                     "mid_channels":64,
                                     "out_channels":64,
                                     "depth_channels":100},
                 depth_refinement_config={"img_depth_norm":'p1',
                                          "pts_depth_norm":'p1',
                                           "merge_depth_norm":'p2'}
                 ):
        super(BiFusionModule,self).__init__()
        self.voxel_size=voxel_size
        self.point_cloud_range=point_cloud_range
        self.point_align_module=PointAlignModule(**point_align_config)
        self.depth_refinement_module=DepthRefinementModule(**depth_refinement_config)
    
    
    """
    valid_voxel_coords(torch.Tensor):(B*N,4)
    valid_voxel_features(torch.Tensor):(B,N,C)
    multi_scale_img_features(list[torch.Tensor]):[(B,num_camera,C,H,W)]*num_level fpn输出的特征图
    img_metas(list[dict]):camera_parameters
    """
    def forward(self,
                img_feats,
                voxel_feats,
                depth_feats,
                voxel_coords,
                voxel_centroids,
                img_metas):
        sample_img_feats,point2depth=self.point_align_module(img_feats,voxel_feats,voxel_coords,voxel_centroids,img_metas,self.voxel_size,self.point_cloud_range)
        updated_depth_feats=self.depth_refinement_module(depth_feats,point2depth)
        return sample_img_feats,updated_depth_feats
         
        