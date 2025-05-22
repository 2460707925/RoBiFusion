import torch

class RoundGridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, grid):
        B, C, H, W = z.shape
        _, N, _ = grid.shape
        device = z.device

        round_grid = grid.round()   
        int_grid = torch.floor(grid)
        diff = grid - int_grid
        weight = torch.abs(1 - 2 * diff[..., 0]) * torch.abs(1 - 2 * diff[..., 1]) 

        round_x = round_grid[..., 0].clamp(0, W - 1)
        round_y = round_grid[..., 1].clamp(0, H - 1)
        round_grid_index = (round_y * W + round_x).long()  

        z_flat = z.view(B, C, H * W)
        index = round_grid_index.unsqueeze(1).expand(-1, C, -1)
        output = z_flat.gather(2, index) 
        output = output * weight.unsqueeze(1) 
        ctx.save_for_backward(round_grid_index, weight, torch.tensor([B, C, H, W], device=device))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        round_grid_index, weight, shape_tensor = ctx.saved_tensors
        B, C, H, W = shape_tensor.tolist()
        _, _, N = grad_output.shape
        device = grad_output.device

        grad_output_weighted = grad_output * weight.unsqueeze(1) 
        grad_z_flat = torch.zeros(B, C, H * W, device=device)
        index = round_grid_index.unsqueeze(1).expand(-1, C, -1) 
        grad_z_flat.scatter_add_(2, index, grad_output_weighted)

        grad_z = grad_z_flat.view(B, C, H, W)

        return grad_z, None
