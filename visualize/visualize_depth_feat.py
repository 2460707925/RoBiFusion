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

from visualize.visualize_hook import VisualFeaturesHooks

def visualize_depth_feat(self, module,input,output):
    context_feats,depth_feats,img_metas=input
    depth_feats=depth_feats.cpu().detach().numpy()
    plt.figure(figsize=(10, 10))
    for i in range(depth_feats.shape[0]):
        plt.subplot(3, 2, i + 1)
        depth_feat=depth_feats[i]
        x,y=depth_feat.shape[-2:]
        plt.plot(depth_feats[i,x//2,y//2])
        # 设置 x 轴和 y 轴的标签，以及图的标题
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(f'Depth Feature {i + 1}')
    np.save(f'{self.save_dir}/depth_feat.npy',depth_feats)
    plt.savefig(f"{self.save_dir}/depth_feat.png")

    
def test_hook():
    config='configs/robifusion/robifusion_nusc_oc.py'
    checkpoint_file='work_dirs/robifusion_nusc_oc/latest.pth'
    save_dir='visualize/visual_feats/camera_depth_feats'
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
    model.img_bev_encoder.register_forward_hook(visual_feats_hooks.visualize_depth_feat_hook)
    
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
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

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
            
        break
    return results
    
if __name__=="__main__":
    test_hook()   
        