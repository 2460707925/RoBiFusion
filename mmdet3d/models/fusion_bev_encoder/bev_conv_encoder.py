from typing import List

import torch 
import torch.nn as nn

from ..builder import FUSION_BEV_ENCODER

@FUSION_BEV_ENCODER.register_module()
class BEVConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, 
                 out_channels: int
                 ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, pts_bev_feats, img_bev_feats) -> torch.Tensor:
        if pts_bev_feats is not None and img_bev_feats is not None:
            inputs=[pts_bev_feats,img_bev_feats]
        elif pts_bev_feats is not None and img_bev_feats is None:
            inputs=[pts_bev_feats]
        elif pts_bev_feats is None and img_bev_feats is not None:
            inputs=[img_bev_feats]
        else:
            raise ValueError("pts_bev_feats and img_bev_feats is both None")
        return super().forward(torch.cat(inputs, dim=1))
    
@FUSION_BEV_ENCODER.register_module()
class BEVConvFuser_Drop(nn.Sequential):
    def __init__(self, in_channels: int, 
                 out_channels: int,
                 is_train:bool=False,
                 lidar_drop:float=0.05,
                 camera_drop:float=0.05,
                 ) -> None:
        self.is_train=is_train
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lidar_drop=lidar_drop
        self.camera_drop=camera_drop
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, pts_bev_feats, img_bev_feats) -> torch.Tensor:
        if pts_bev_feats is not None and img_bev_feats is not None:
            # 训练模式中随机的失去一个模态的特征信息
            if self.is_train:
                lidar_drop_flag=torch.rand(1).item()<self.lidar_drop
                camera_drop_flag=torch.rand(1).item()<self.camera_drop
                # 训练时随机替换输入的feats
                if lidar_drop_flag^camera_drop_flag:
                    if lidar_drop_flag:
                        pts_bev_feats=torch.zeros_like(pts_bev_feats,requires_grad=False)
                    else:
                        img_bev_feats=torch.zeros_like(img_bev_feats,requires_grad=False)
            inputs=[pts_bev_feats,img_bev_feats]
        elif pts_bev_feats is not None and img_bev_feats is None:
            inputs=[pts_bev_feats]
        elif pts_bev_feats is None and img_bev_feats is not None:
            inputs=[img_bev_feats]
        else:
            raise ValueError("pts_bev_feats and img_bev_feats is both None")
        return super().forward(torch.cat(inputs, dim=1))