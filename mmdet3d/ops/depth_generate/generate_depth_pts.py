import torch
import torch.nn as nn
from torch.autograd import Function
from . import depth_generate_ext

class GenerateDepthFunctionPTS(Function):
    # sampling_positions:(M,2)
    # sampling_depth_features:(M,D)
    # featuer_shape:H,W
    
    # updated_depth_features:(H,W,D)
    @staticmethod
    def forward(ctx,
                sampling_positions,
                sampling_depth_features,
                feature_shape):
        device=sampling_depth_features.device
        height,width=feature_shape
        sampling_num,depth_dim=sampling_depth_features.shape
        shape_value=torch.tensor([sampling_num,height,width,depth_dim],dtype=torch.int32,requires_grad=False,device=device)
        updated_depth_features = torch.zeros((height,width,depth_dim),dtype=torch.float32,requires_grad=True,device=device)

        sampling_indexes = torch.zeros((sampling_num,4), dtype=torch.int32, device=device,requires_grad=False)
        sampling_confs = torch.zeros((sampling_num,4),dtype=torch.float32,device=device,requires_grad=False)
        
        if not sampling_depth_features.is_contiguous():
            sampling_depth_features=sampling_depth_features.contiguous()
        
        positions_x=sampling_positions[:,0].contiguous()
        positions_y=sampling_positions[:,1].contiguous()
        
        sampling_positions=sampling_positions.detach()
        depth_generate_ext.torch_launch_generate_depth(positions_x,
                                                             positions_y,
                                                             sampling_depth_features,
                                                             updated_depth_features,
                                                             sampling_indexes,
                                                             sampling_confs,
                                                             shape_value)
        ctx.save_for_backward(shape_value,sampling_indexes,sampling_confs)
        return updated_depth_features

    # grad_output:(H,W,D)        
    @staticmethod
    def backward(ctx,
                 grad_output):
        device=grad_output.device
        shape_value,sampling_indexes,sampling_confs=ctx.saved_variables
        sampling_num=shape_value[0]
        depth_dim=shape_value[3]
        grad_sampling_depth_features=torch.zeros((sampling_num,depth_dim),dtype=torch.float32,device=device,requires_grad=False)
        
        # 对输入的tensor进行连续操作
        if not grad_output.is_contiguous():
            grad_output=grad_output.contiguous()
        
        depth_generate_ext.torch_backward_generate_depth(grad_output,
                                                               grad_sampling_depth_features,
                                                               sampling_indexes,
                                                               sampling_confs,
                                                               shape_value)
        
        return None, grad_sampling_depth_features, None
        


class GeneratePtsDepthMoudle(nn.Module):
    def __init__(self):
        super(GeneratePtsDepthMoudle, self).__init__()

    def forward(self,
                sampling_positions,
                sampling_depth_features,
                feature_shape):
        depth_image_pts=GenerateDepthFunctionPTS.apply(sampling_positions,
                                                       sampling_depth_features,
                                                       feature_shape)
        return depth_image_pts
    