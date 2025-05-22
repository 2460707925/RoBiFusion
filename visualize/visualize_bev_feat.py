import torch 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

if __name__=="__main__":
    bev_dir='visualize/bev_feats'
    saved_dir='visualize/visual_feats/bev_feats'
    saved_bev_feats_src=torch.load(f'{bev_dir}/saved_bev_feats_src.pth')
    saved_bev_feats_tgt=torch.load(f'{bev_dir}/saved_bev_feats_tgt.pth')
    saved_bev_feats_tgt1=torch.load(f'{bev_dir}1/saved_bev_feats_tgt.pth')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))
    fig.suptitle('Camera BEV Features')
    fig.subplots_adjust(top=0.95)
    for i in range(len(saved_bev_feats_src)):
        src_bev_feat=saved_bev_feats_src[i]
        tgt_bev_feat=saved_bev_feats_tgt[i]
        tgt_bev_feat1=saved_bev_feats_tgt1[i]
        src_bev_feat_var=torch.sqrt(torch.var(src_bev_feat.squeeze(0),dim=0))
        src_bev_feat_var=src_bev_feat_var.cpu().detach().numpy()
        tgt_bev_feat_var=torch.var(tgt_bev_feat.squeeze(0),dim=0)/9
        tgt_bev_feat_var1=torch.var(tgt_bev_feat1.squeeze(0),dim=0)/9
        tgt_bev_feat_var=torch.max(tgt_bev_feat_var,tgt_bev_feat_var1)
        tgt_bev_feat_var=tgt_bev_feat_var.cpu().detach().numpy()
        im1=axs[0].imshow(src_bev_feat_var, interpolation='bilinear')
        axs[0].set_xlabel('BEVFusion')
        im2=axs[1].imshow(tgt_bev_feat_var, interpolation='bilinear')
        axs[1].set_xlabel('RoBiFusion')
        fig.subplots_adjust(right=0.9)
        if i==0:
            l = 0.92
            b = 0.14
            w = 0.015
            h = 0.78
            rect = [l,b,w,h] 
            cbar_ax = fig.add_axes(rect) 
            cb = plt.colorbar(im2, cax=cbar_ax, orientation='vertical')
        fig.savefig(f'{saved_dir}/{i}_src_bev_feat.png')
    