import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from ..builder import LOSSES

@LOSSES.register_module()
class DepthLoss(nn.Module):
    def __init__(self,depth_loss_factor,downsample_factor,dbound):
        super(DepthLoss,self).__init__()
        self.depth_loss_factor=depth_loss_factor
        self.downsample_factor=downsample_factor
        self.dbound=dbound
        self.depth_channels=int((dbound[1]-dbound[0])//dbound[2])

    def forward(self, depth_labels, depth_preds):

        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_channels)
        depth_preds = torch.clamp(depth_preds, min=0.01)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))
        return self.depth_loss_factor * depth_loss

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
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                    W // self.downsample_factor)

        gt_depths = (gt_depths - (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),num_classes=self.depth_channels + 1).view(-1, self.depth_channels + 1)[:, 1:]

        return gt_depths.float()