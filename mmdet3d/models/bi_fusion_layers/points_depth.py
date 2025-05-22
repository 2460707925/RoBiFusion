import torch 
import torch.nn as nn
import torch.nn.functional as F 

from mmdet3d.ops.depth_generate import GeneratePtsDepthMoudle


class GenerateDepthFeature_Pts(nn.Module):
    def __init__(self,
                 sample_range=[-6,6],
                 depth_dim=256):
        super(GenerateDepthFeature_Pts,self).__init__()
        self.depth_dim=depth_dim
        d_coors1=torch.linspace(sample_range[0],0,depth_dim)
        d_coors2=torch.linspace(0,sample_range[1],depth_dim)
        self.d_coors=torch.cat((d_coors1,d_coors2))
        self.mask = torch.arange(len(self.d_coors)).unsqueeze(0)
        
        self.gen_depth_feature_module=GeneratePtsDepthMoudle()
        
        
        
    def forward(self,
                pts_pos,
                pts_depth,
                pts_conf,
                feature_shape):
        # 采样生成pts的depth的概率分布
        from time import time 
        t1=time()
        device=pts_depth.device
        pts_num=pts_pos.shape[0]
        mask=self.mask.expand(pts_num, -1).to(device=device)
        mask=(mask>=(self.depth_dim-1-pts_depth)) & (mask<=(2*self.depth_dim-2-pts_depth))
        # x:pts_num,depth_dim
        x=self.d_coors.expand(pts_num,-1)[mask].view(pts_num,-1).to(device=device)
        dist=torch.distributions.Normal(loc=0, scale=pts_conf)
        pts_depth_probs=F.normalize(torch.exp(dist.log_prob(x)),p=2,dim=-1)
        if not pts_pos.is_cuda:
            pts_pos=pts_pos.cuda()

        if not pts_depth_probs.is_cuda:
            pts_depth_probs=pts_depth_probs.cuda()
            
        pts_depth_feature_map=self.gen_depth_feature_module(pts_pos,pts_depth_probs,feature_shape)
        return pts_depth_feature_map
        

        
if __name__=="__main__":
    depth_dim=100
    voxel_num=100
    feature_shape=[10,10]
    generate_depth_module=GenerateDepthFeature_Pts(depth_dim=depth_dim)
    pts_ops=torch.randint(low=0,high=feature_shape[0]-1,size=(voxel_num,2))
    d1=torch.randint(low=0,high=depth_dim-1,size=(voxel_num,1))
    depth=torch.rand((voxel_num)).view(voxel_num,1)
    pts_depth=generate_depth_module(pts_ops,d1,depth,feature_shape)
    
    import matplotlib.pyplot as plt
    plt.ion()
    x=torch.linspace(0,depth_dim-1,depth_dim)
    plt.scatter(x, pts_depth[0][0].cpu().detach().numpy(),c='green',label='show')
    plt.show()
    
    
    # fig, ax = plt.subplots()
    # pts_depth_add=pts_depth[0]+pts_depth[1]+pts_depth[2]
    # pts_depth_add=F.normalize(pts_depth_add,p=1,dim=0)
    # pts_depth_src=F.normalize(pts_depth[3],p=1,dim=0)
    
    # pts_depth_mul=pts_depth_src*pts_depth_add
    # pts_depth_mul=F.normalize(pts_depth_mul,p=2,dim=0)
    
    # ax.scatter(x, pts_depth_src.detach().numpy(),c='green',label='src')
    # ax.scatter(x, pts_depth_add.detach().numpy(),c='red',label='add')
    # ax.scatter(x, pts_depth_mul.detach().numpy(), c='blue', label='mul')
    # plt.show()
        
        