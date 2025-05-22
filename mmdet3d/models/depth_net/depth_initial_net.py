import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

from mmdet.models.backbones.resnet import BasicBlock

from mmcv.runner import force_fp32
from mmcv.cnn import build_conv_layer
from ..builder import DEPTH_NET

# 使用depth consistency
@DEPTH_NET.register_module()
class DepthInitialModule(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 num_embedding_parameters=27
                 ):
        super(DepthInitialModule,self).__init__()
        self.depth_channels=depth_channels
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        # context_channels
        self.context_conv=nn.Conv2d(
            mid_channels,
            context_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.context_mlp = Mlp(num_embedding_parameters, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware    
    
        # Camera Parameters Embedding,输入的内参函数维度为26维
        self.num_embedding_parameters=num_embedding_parameters        
        self.bn = nn.BatchNorm1d(self.num_embedding_parameters)
        
        self.depth_mlp = Mlp(self.num_embedding_parameters, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    @force_fp32()
    def get_mlp_input(self,batch_size,num_cams,img_metas,device):
        intrins=[]
        ida=[]
        sensor2ego=[]
        bda=[]
        for ibatch in range(batch_size):
            img_meta=img_metas[ibatch]
            intrins.append(img_meta['camera_intrinsics'].data[:,:3,:3])
            ida.append(img_meta["img_aug_matrix"].data[:, ...])
            sensor2ego.append(img_meta["camera2ego"].data[:, :3, :])
            bda.append(img_meta["lidar_aug_matrix"].data.view(1, 4, 4).repeat(num_cams, 1, 1))
        
        intrins=torch.stack(intrins,dim=0)
        ida=torch.stack(ida,dim=0)
        sensor2ego=torch.stack(sensor2ego,dim=0)
        bda=torch.stack(bda,dim=0)
        
        # If exporting, cache the MLP input, since it's based on 
        # intrinsics and data augmentation, which are constant at inference time. 
        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, ..., 0, 0],
                        intrins[:, ..., 1, 1],
                        intrins[:, ..., 0, 2],
                        intrins[:, ..., 1, 2],
                        ida[:, ..., 0, 0],
                        ida[:, ..., 0, 1],
                        ida[:, ..., 0, 3],
                        ida[:, ..., 1, 0],
                        ida[:, ..., 1, 1],
                        ida[:, ..., 1, 3],
                        bda[:, ..., 0, 0],
                        bda[:, ..., 0, 1],
                        bda[:, ..., 1, 0],
                        bda[:, ..., 1, 1],
                        bda[:, ..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(batch_size, num_cams, -1),
            ],
            -1,
        ).to(device=device)
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        return mlp_input

    # 输入特征图，并进行参数的设置
    def forward(self, img_features , img_metas):
        batch_size=len(img_metas)
        num_camera=img_features.shape[0]//batch_size
        _,feature_channels,feature_height,feature_width=img_features.shape
        
        mlp_input=self.get_mlp_input(batch_size,num_camera,img_metas,img_features.device)
        img_features=img_features.reshape(-1,feature_channels,feature_height,feature_width)
        mid_features = self.reduce_conv(img_features) # (B*num_camera,C,H,W)->(B*num_camera,mid_channels,H,W)
        
        # 提取语义特征
        context_se=self.context_mlp(mlp_input)[...,None,None]
        context=self.context_se(mid_features,context_se)
        context=self.context_conv(context)
        
        # depth语义特征
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth_features = self.depth_se(mid_features, depth_se)
        depth_features = self.depth_conv(depth_features).reshape(batch_size*num_camera,self.depth_channels,
                                                                                feature_height,feature_width)
        
        # depth_features = torch.nan_to_num(depth_features)
        depth_features=depth_features.float()
        depth_features = depth_features.softmax(dim=1)
        return depth_features,context


# 两层的mlp层
class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
# 对输入的Layer添加sigmoid函数进行激活操作
class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# 进行多个维度的空洞卷积，然后合在一起
class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                

