import torch.nn as nn 
import torch.nn.functional as F



class DepthRefinementModule(nn.Module):
    def __init__(self,
                 img_depth_norm='p1',
                 pts_depth_norm='p1',
                 merge_depth_norm='p2'):
        super(DepthRefinementModule,self).__init__()
        norm_type_dict={'p1':1,'p2':2}
        self.img_depth_norm=norm_type_dict[img_depth_norm]
        self.pts_depth_norm=norm_type_dict[pts_depth_norm]
        self.merge_depth_norm=norm_type_dict[merge_depth_norm]

    def forward(self,
                init_depth_images,
                pts_depth_images):
        assert len(init_depth_images.shape)==4 and len(pts_depth_images.shape)==4,"depth image shape (B*N,D,H,W)"
        # normalized_init_depth_images=F.normalize(init_depth_images,dim=1,p=self.img_depth_norm)
        # normalized_pts_depth_images=F.normalize(pts_depth_images,dim=1,p=self.pts_depth_norm)
        updated_depth_images=init_depth_images*pts_depth_images
        
        normalized_updated_depth_images=F.normalize(updated_depth_images,dim=1,p=1)
        return normalized_updated_depth_images