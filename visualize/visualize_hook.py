import os 
import cv2
import torch 
import torch.nn.functional as F 

import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import (load_checkpoint,wrap_fp16_model)
from mmdet3d.datasets import build_dataset,build_dataloader
from mmdet3d.models import build_detector

class VisualFeaturesHooks:
    def __init__(self,save_dir):
        self.save_dir=save_dir
        
    def after_depth_refinement_hook(self,module,input,output):
        init_depth_images,pts_depth_images=input
        normalized_init_depth_images=F.normalize(init_depth_images,dim=1,p=module.img_depth_norm)
        normalized_pts_depth_images=F.normalize(pts_depth_images,dim=1,p=module.pts_depth_norm)
        updated_depth_images=normalized_init_depth_images*normalized_pts_depth_images
        normalized_updated_depth_images=F.normalize(updated_depth_images,dim=1,p=module.merge_depth_norm)
        
        
        torch.save(init_depth_images,f'{self.save_dir}/img_depth_feats.pth')
        torch.save(pts_depth_images,f'{self.save_dir}/pts_depth_feats.pth')
        torch.save(normalized_updated_depth_images,f'{self.save_dir}/fuse_depth_feats.pth')
                
                
    def visualize_bev_feats_hook(self,module,input,output):
        module_name=type(module).__name__.lower()
        torch.save(output,f"{self.save_dir}/{module_name}_bev_feats.pth")

    
        
def test_hook():
    config='configs/robifusion/robifusion_nusc_lc_without_gtdepth.py'
    checkpoint_file='work_dirs/robifusion_nusc_lc_without_gtdepth/latest.pth'
    save_dir='./visualize/visual_feats/depth_and_bev_feats'
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
        shuffle=True)
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # 添加hook
    visual_feats_hooks=VisualFeaturesHooks(save_dir=save_dir)
    model.bi_fusion_layer.depth_refinement_module.register_forward_hook(visual_feats_hooks.after_depth_refinement_hook)
    model.img_bev_encoder.register_forward_hook(visual_feats_hooks.visualize_bev_feats_hook)
    model.pts_middle_encoder.register_forward_hook(visual_feats_hooks.visualize_bev_feats_hook)
    model.fuse_bev_encoder.register_forward_hook(visual_feats_hooks.visualize_bev_feats_hook)
    
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
        for img_file in img_files:
            img=cv2.imread(img_file)
            img=cv2.resize(img,(800,448))
            imgs.append(img)
        torch.save(imgs,f"{save_dir}/imgs.pth")
        torch.save(data,f'{save_dir}/data.pth')
        
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