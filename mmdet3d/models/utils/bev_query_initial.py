import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from mmcv.cnn import ConvModule, build_conv_layer

class General_BEV_Query_Initialization(nn.Module):
    def __init__(self,
                 in_channels,
                 bev_pos,
                 num_proposals,
                 bias="auto",
                 num_classes=4,
                 nms_kernel_size=1,
                 train_cfg=None,
                 test_cfg=None):
        super(General_BEV_Query_Initialization,self).__init__()
        self.bev_pos = bev_pos
        self.num_proposals=num_proposals
        self.num_classes=num_classes
        self.nms_kernel_size=nms_kernel_size
        self.train_cfg=train_cfg
        self.test_cfg=test_cfg
        
        layers = []
        layers.append(
            ConvModule(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type="Conv2d"),
                norm_cfg=dict(type="BN2d"),
            )
        )
        layers.append(
            build_conv_layer(
                dict(type="Conv2d"),
                in_channels,
                num_classes,
                kernel_size=3,
                padding=1,
                bias=bias,
            )
        )
        self.heatmap_head = nn.Sequential(*layers)
        self.class_encoding = nn.Conv1d(num_classes, in_channels, 1)
        
    def forward(self,bev_feats,history_query=None,metas=None):
        batch_size=bev_feats.shape[0]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(bev_feats.device)
        bev_feats_flatten=bev_feats.view(batch_size, bev_feats.shape[1], -1)  # [BS, C, H*W]
        dense_heatmap = self.heatmap_head(bev_feats)

        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d( heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        if padding==0:
            local_max=local_max_inner
        else:
            local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        ## for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg["dataset"] == "nuScenes":
            local_max[:, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg["dataset"] == "Waymo":  # for Pedestrian & Cyclist in Waymo
            local_max[:, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[:, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # top #num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., : self.num_proposals]
        
        # query_labels
        query_labels = torch.div(top_proposals,heatmap.shape[-1],rounding_mode='trunc')
        # query_labels = top_proposals // heatmap.shape[-1]
    
        # query_pos
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_pos = bev_pos.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),dim=1)
        
        # query_feat
        query_feat= bev_feats_flatten.gather(index=top_proposals_index[:, None, :].expand(-1, bev_feats_flatten.shape[1], -1),dim=-1)
        # add category embedding
        one_hot = F.one_hot(query_labels, num_classes=self.num_classes).permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding
        
        query_heatmap_score=heatmap.gather( index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1) 
        
        if history_query is not None:
            raise NotImplementedError
        
        return query_feat,query_pos,query_labels,query_heatmap_score,dense_heatmap
    
        
    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base