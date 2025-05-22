import torch 
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class GTDepth:
    def __init__(self, keyframe_only=False):
        self.keyframe_only = keyframe_only 

    def __call__(self, data):
        img_aug_matrix = data['img_aug_matrix'].data 
        if 'lidar_aug_matrix' in data.keys():
            bev_aug_matrix = data['lidar_aug_matrix'].data
        else:
            bev_aug_matrix = torch.eye(4)
            
        lidar2image = data['lidar2image'].data

        points = data['points'].data 
        img = data['img'].data

        if self.keyframe_only:
            points = points[points[:, 4] == 0]

        batch_size = len(points)
        depth = torch.zeros(img.shape[0], *img.shape[-2:]) #.to(points[0].device)

        # for b in range(batch_size):
        cur_coords = points[:, :3]

        # inverse aug
        cur_coords -= bev_aug_matrix[:3, 3]
        cur_coords = torch.inverse(bev_aug_matrix[:3, :3]).matmul(cur_coords.transpose(1, 0))
        # lidar2image
        cur_coords = lidar2image[:, :3, :3].matmul(cur_coords)
        cur_coords += lidar2image[:, :3, 3].reshape(-1, 3, 1)
        # get 2d coords
        dist = cur_coords[:, 2, :]
        cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
        cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

        # imgaug
        cur_coords = img_aug_matrix[:, :3, :3].matmul(cur_coords)
        cur_coords += img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
        cur_coords = cur_coords[:, :2, :].transpose(1, 2)

        # normalize coords for grid sample
        cur_coords = cur_coords[..., [1, 0]]

        on_img = (
            (cur_coords[..., 0] < img.shape[2])
            & (cur_coords[..., 0] >= 0)
            & (cur_coords[..., 1] < img.shape[3])
            & (cur_coords[..., 1] >= 0)
        )
        for c in range(on_img.shape[0]):
            masked_coords = cur_coords[c, on_img[c]].long()
            masked_dist = dist[c, on_img[c]]
            depth[c, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        data['gt_depth'] = depth 
        
        return data