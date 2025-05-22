import os 
import cv2
import numpy as np
import torch 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
from types import MethodType

import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import (load_checkpoint,wrap_fp16_model)
from mmdet3d.datasets import build_dataset,build_dataloader
from mmdet3d.models import build_detector
from mmdet3d.models.bi_fusion_layers.depth_refinement import DepthRefinementModule

from visualize.visualize_hook import VisualFeaturesHooks
from visualize.utils import my_decorator

img_depth_feat=None
gt_depth_feat=None

def visualize_depth_feat(self, module,input,output):
    global img_depth_feat
    context_feats,depth_feats,img_metas=input
    depth_feats=depth_feats.permute(0,2,3,1)
    img_depth_feat=depth_feats.clone()
    depth_feats=F.normalize(depth_feats,dim=1,p=1)
    depth_feats=depth_feats.cpu().detach().numpy()
    np.save(f'{self.save_dir}/depth_feat.npy',depth_feats)
    
# 注册在forward_test前
def visualize_gt_depth_feat(self, *args, **kwargs):
    global gt_depth_feat
    gt_depths=kwargs['gt_depth']
    module=args[0]
    depth_module=module.loss_depth  
    B, N, H, W = gt_depths.shape
    h,w=H // depth_module.downsample_factor,W // depth_module.downsample_factor     
    depth_labels_max,depth_labels_min=get_downsampled_gt_depth(depth_module,gt_depths)
    depth_labels_max=depth_labels_max.reshape(B,N,h,w,-1)
    depth_labels_min=depth_labels_min.reshape(B,N,h,w,-1)
    output_shape=depth_labels_max.shape
    depth_labels_max=torch.argmax(depth_labels_max,axis=-1)
    depth_labels_min=torch.argmax(depth_labels_min,axis=-1)
    depth_feature_max=generate_depth_feature(output_shape,depth_labels_max).view(B*N,h,w,-1).permute(0,3,1,2)
    depth_feature_min=generate_depth_feature(output_shape,depth_labels_min).view(B*N,h,w,-1).permute(0,3,1,2)
    depth_labels=torch.cat((depth_labels_max.unsqueeze(-1),depth_labels_min.unsqueeze(-1)),dim=-1)
    depth_feature=F.normalize(depth_feature_min+depth_feature_max,dim=1,p=1).permute(0,2,3,1)
    gt_depth_feat=depth_feature.clone()
    depth_feature=depth_feature.detach().cpu().numpy()
    depth_labels=depth_labels.view(B*N,h,w,-1).detach().cpu().numpy()
    np.save(f'{self.save_dir}/gt_depth_labels.npy',depth_labels)
    np.save(f'{self.save_dir}/gt_depth.npy',depth_feature)

def get_downsampled_gt_depth(self, gt_depths):
    """
    Input:
        gt_depths: [B, N, H, W]
    Output:
        gt_depths: [B*N*h*w, d]
    """
    B, N, H, W = gt_depths.shape
    gt_depths = gt_depths.view(
        B * N,
        H // self.downsample_factor,
        self.downsample_factor,
        W // self.downsample_factor,
        self.downsample_factor,
        1,
    )
    gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
    gt_depths = gt_depths.view(
        -1, self.downsample_factor * self.downsample_factor)
    gt_depths_max=torch.max(gt_depths,dim=-1).values
    
    gt_depths_tmp = torch.where(gt_depths == 0.0,
                                1e5 * torch.ones_like(gt_depths),
                                gt_depths)
    gt_depths_min = torch.min(gt_depths_tmp, dim=-1).values
    
    gt_depths_max=gt_depths_max.view(B * N, H // self.downsample_factor, W // self.downsample_factor)
    gt_depths_min = gt_depths_min.view(B * N, H // self.downsample_factor, W // self.downsample_factor)
    
    gt_depths_max = (gt_depths_max - (self.dbound[0] - self.dbound[2])) / self.dbound[2]
    gt_depths_min = (gt_depths_min - (self.dbound[0] - self.dbound[2])) / self.dbound[2]
    
    gt_depths_max = torch.where(
        (gt_depths_max < self.depth_channels + 1) & (gt_depths_max >= 0.0),
        gt_depths_max, torch.zeros_like(gt_depths_max))

    gt_depths_min = torch.where(
        (gt_depths_min < self.depth_channels + 1) & (gt_depths_min >= 0.0),
        gt_depths_min, torch.zeros_like(gt_depths_min))

    gt_depths_max = F.one_hot(gt_depths_max.long(),num_classes=self.depth_channels + 1).view(-1, self.depth_channels + 1)[:, 1:]
    gt_depths_min = F.one_hot(gt_depths_min.long(),num_classes=self.depth_channels + 1).view(-1, self.depth_channels + 1)[:, 1:]

    return gt_depths_max.float(),gt_depths_min.float()
    
def generate_depth_feature(output_shape,depth_index):
    device=depth_index.device
    B,N,h,w,depth_dim=output_shape
    sample_range=[-6,6]
    d_coors1=torch.linspace(sample_range[0],0,depth_dim)
    d_coors2=torch.linspace(0,sample_range[1],depth_dim)
    d_coors=torch.cat((d_coors1,d_coors2))
    mask = torch.arange(len(d_coors)).unsqueeze(0)
    mask = mask.expand(B,N,h,w,-1).to(device=device)
    mask=(mask>=(depth_dim-1-depth_index).unsqueeze(-1)) & (mask<=(2*depth_dim-2-depth_index).unsqueeze(-1))
    x=d_coors.expand(B,N,h,w,-1)[mask].view(B,N,h,w,-1).to(device=device)
    pts_conf=torch.ones(x.shape[:-1]).unsqueeze(-1).to(device=device)/10
    dist=torch.distributions.Normal(loc=0, scale=pts_conf)
    pts_depth_probs=F.normalize(torch.exp(dist.log_prob(x)),p=2,dim=-1)
    return pts_depth_probs
    
def test_hook():
    config='configs/robifusion/robifusion_nusc_oc.py'
    checkpoint_file='work_dirs/robifusion_nusc_oc/latest.pth'
    save_dir='visualize/visual_feats/camera_depth_feats'
    depth_refine_module=DepthRefinementModule()
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cfg=Config.fromfile(config)
    cfg.model.pretrained = None
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # 添加hook
    visual_feats_hooks=VisualFeaturesHooks(save_dir=save_dir)
    visual_feats_hooks.visualize_depth_feat_hook=MethodType(visualize_depth_feat,visual_feats_hooks)
    visual_feats_hooks.visualize_gt_depth_feat_hook=MethodType(visualize_gt_depth_feat,visual_feats_hooks)
    new_forward_test=my_decorator(model.forward_test,visual_feats_hooks.visualize_gt_depth_feat_hook)
    model.img_bev_encoder.register_forward_hook(visual_feats_hooks.visualize_depth_feat_hook)
    model.forward_test=MethodType(new_forward_test,model)
    
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        if i ==10:
            # 保存图像和点云地址
            img_files=data['img_metas'].data[0][0]['img_filename']
            imgs=[]
            for i,img_file in enumerate(img_files):
                img=cv2.imread(img_file)
                img=cv2.resize(img,(800,448))
                imgs.append(img)
                cv2.imwrite(f'{save_dir}/{i}.jpg', img)
            
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)

            results.extend(result)
            
            fuse_depth_feat=depth_refine_module(img_depth_feat,gt_depth_feat).cpu().detach().numpy()
            np.save(f'{save_dir}/fuse_depth_feat.npy',fuse_depth_feat)

            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()
                
            break
    return results
    
if __name__=="__main__":
    test_hook()   
        