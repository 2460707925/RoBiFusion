import copy
import numpy as np
import torch
from mmcv.ops import boxes_iou3d,nms_bev
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr)
from mmdet3d.models import builder
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import clip_sigmoid
from mmdet.core import build_bbox_coder, multi_apply




@HEADS.register_module()
class DSVT_CenterHead(nn.Module):
    """CenterHead for CenterPoint.

    Args:
        mode (str): Mode of the head. Default: '3d'.
        in_channels (list[int] | int): Channels of the input feature map.
            Default: [128].
        tasks (list[dict]): Task information including class number
            and class names. Default: None.
        dataset (str): Name of the dataset. Default: 'nuscenes'.
        weight (float): Weight for location loss. Default: 0.25.
        code_weights (list[int]): Code weights for location loss. Default: [].
        common_heads (dict): Conv information for common heads.
            Default: dict().
        loss_cls (dict): Config of classification loss function.
            Default: dict(type='GaussianFocalLoss', reduction='mean').
        loss_bbox (dict): Config of regression loss function.
            Default: dict(type='L1Loss', reduction='none').
        separate_head (dict): Config of separate head. Default: dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3)
        share_conv_channel (int): Output channels for share_conv_layer.
            Default: 64.
        num_heatmap_convs (int): Number of conv layers for heatmap conv layer.
            Default: 2.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels=[128],
                 tasks=None,
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 common_heads=dict(),
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(
                     type='L1Loss', reduction='none', loss_weight=0.25),
                 loss_iou=None,
                 loss_reg_iou=None,
                 separate_head=dict(
                     type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64,
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 norm_bbox=True):
        super(DSVT_CenterHead, self).__init__()

        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.norm_bbox = norm_bbox

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou) if loss_iou is not None else None
        self.loss_iou_reg = build_loss(loss_reg_iou) if loss_reg_iou is not None else None
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_anchor_per_locs = [n for n in num_classes]
        self.fp16_enabled = False

        # a shared convolution
        self.shared_conv = ConvModule(
            in_channels,
            share_conv_channel,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias)

        self.task_heads = nn.ModuleList()

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(num_cls, num_heatmap_convs)))
            separate_head.update(
                in_channels=share_conv_channel, heads=heads, num_cls=num_cls)
            self.task_heads.append(builder.build_head(separate_head))

    def init_weights(self):
        """Initialize weights."""
        for task_head in self.task_heads:
            task_head.init_weights()

    def forward_single(self, x):
        """Forward function for CenterPoint.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            list[dict]: Output results for tasks.
        """
        ret_dicts = []

        x = self.shared_conv(x)

        for task in self.task_heads:
            ret_dicts.append(task(x))

        return ret_dicts

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        res = multi_apply(self.forward_single, [feats])
        return res

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor): Mask of the feature map with the shape
                of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def get_targets(self, gt_bboxes_3d, gt_labels_3d):
        """Generate targets.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including \
                    the following results in order.

                    - list[torch.Tensor]: Heatmap scores.
                    - list[torch.Tensor]: Ground truth boxes.
                    - list[torch.Tensor]: Indexes indicating the \
                        position of the valid boxes.
                    - list[torch.Tensor]: Masks indicating which \
                        boxes are valid.
        """
        heatmaps, anno_boxes, inds, masks, task_gt_bboxes = multi_apply(
            self.get_targets_single, gt_bboxes_3d, gt_labels_3d)
        # transpose heatmaps, because the dimension of tensors in each task is
        # different, we have to use numpy instead of torch to do the transpose.
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # Transpose anno_boxes
        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # Transpose inds
        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(inds_) for inds_ in inds]
        # Transpose inds
        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(masks_) for masks_ in masks]
        task_gt_bboxes = list(map(list, zip(*task_gt_bboxes)))
        return heatmaps, anno_boxes, inds, masks ,task_gt_bboxes

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]))

            anno_box = gt_bboxes_3d.new_zeros((max_objs, 10),
                                              dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue
                    
                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]
                    mask[new_idx] = 1
                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])

                    ind[new_idx] = y * feature_map_size[0] + x
                    # TODO: support other outdoor dataset
                    vx, vy = task_boxes[idx][k][7:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x, y], device=device),
                        z.unsqueeze(0), box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0),
                        vx.unsqueeze(0),
                        vy.unsqueeze(0)
                    ])

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks, task_boxes

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks, task_gt_bboxes = self.get_targets(
            gt_bboxes_3d, gt_labels_3d)
        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]['anno_box'] = torch.cat(
                (preds_dict[0]['reg'], preds_dict[0]['height'],
                 preds_dict[0]['dim'], preds_dict[0]['rot'],
                 preds_dict[0]['vel']),
                dim=1)

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
            
            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
            
            if self.loss_iou_reg is not None:
                batch_box_preds = self._decode_all_preds(
                    pred_dict=preds_dict[0],
                    point_cloud_range=self.train_cfg['point_cloud_range'],
                    voxel_size=self.train_cfg['voxel_size']
                )  # (B, H, W, 7 or 9)

                batch_box_preds_for_iou = batch_box_preds.permute(
                    0, 3, 1, 2)  # (B, 7 or 9, H, W)
                
                loss_dict[f'task{task_id}.loss_reg_iou'] = \
                    self.calc_iou_reg_loss(
                        batch_box_preds=batch_box_preds_for_iou,
                        mask=masks[task_id],
                        ind=ind,
                        gt_boxes=task_gt_bboxes[task_id])
                

                if 'iou' in preds_dict[0]:
                    loss_dict[f'task{task_id}.loss_iou'] = self.calc_iou_loss(
                        iou_preds=preds_dict[0]['iou'],
                        batch_box_preds=batch_box_preds_for_iou.clone().detach(),
                        mask=masks[task_id],
                        ind=ind,
                        gt_boxes=task_gt_bboxes[task_id])

            
        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)
            batch_iou = (preds_dict[0]['iou'] +
                         1) * 0.5 if 'iou' in preds_dict[0] else None

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                iou=batch_iou)
            assert self.test_cfg['nms_type'] in ['circle', 'rotate']
            batch_reg_preds, batch_cls_preds, batch_cls_labels, batch_iou_preds = [], [], [], []  # noqa: E501
            for box in temp:
                batch_reg_preds.append(box['bboxes'])
                batch_cls_preds.append(box['scores'])
                batch_cls_labels.append(box['labels'].long())
                batch_iou_preds.append(box['iou'])
            if self.test_cfg['nms_type'] == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(task_id,num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_iou_preds, batch_cls_labels,
                                             img_metas))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = img_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list

    def get_task_detections(self, task_id, num_class_with_bg, batch_cls_preds,
                            batch_reg_preds, batch_iou_preds, batch_cls_labels,
                            img_metas):
        """Rotate nms for each task.

        Args:
            num_class_with_bg (int): Number of classes for the current task.
            batch_cls_preds (list[torch.Tensor]): Prediction score with the
                shape of [N].
            batch_reg_preds (list[torch.Tensor]): Prediction bbox with the
                shape of [N, 9].
            batch_iou_preds (list[torch.Tensor]): Prediction IoU with the
                shape of [N].
            batch_cls_labels (list[torch.Tensor]): Prediction label with the
                shape of [N].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            list[dict[str: torch.Tensor]]: contains the following keys:

                -bboxes (torch.Tensor): Prediction bboxes after nms with the
                    shape of [N, 9].
                -scores (torch.Tensor): Prediction scores after nms with the
                    shape of [N].
                -labels (torch.Tensor): Prediction labels after nms with the
                    shape of [N].
        """
        predictions_dicts = []
        for i, (box_preds, cls_preds, iou_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_iou_preds,
                    batch_cls_labels)):
            pred_iou = torch.clamp(iou_preds, min=0, max=1.0)
            iou_rectifier = pred_iou.new_tensor(
                self.test_cfg['iou_rectifier'][task_id])
            cls_preds = torch.pow(cls_preds,
                                  1 - iou_rectifier[cls_labels]) * torch.pow(
                                      pred_iou, iou_rectifier[cls_labels])

            # Apply NMS in bird eye view
            # get the highest score per prediction, then apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = cls_preds
                top_labels = torch.zeros(
                    cls_preds.shape[0],
                    device=cls_preds.device,
                    dtype=torch.long)

            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds

            if top_scores.shape[0] != 0:
                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](
                    box_preds[:, :], self.bbox_coder.code_size).bev)

                pre_max_size = self.test_cfg['pre_max_size'][task_id]
                post_max_size = self.test_cfg['post_max_size'][task_id]
                # cls_label_per_task = self.cls_id_mapping_per_task[task_id]
                all_selected_mask = torch.zeros_like(top_labels, dtype=bool)
                all_indices = torch.arange(top_labels.size(0)).to(
                    top_labels.device)
                # Mind this when training on the new coordinate
                # Transform to old mmdet3d coordinate
                boxes_for_nms[:, 4] = (-boxes_for_nms[:, 4] + torch.pi / 2 * 1)
                boxes_for_nms[:, 4] = (boxes_for_nms[:, 4] +
                                       torch.pi) % (2 * torch.pi) - torch.pi

                for i, nms_thr in enumerate(self.test_cfg['nms_thr'][task_id]):
                    label_mask = top_labels == i
                    selected = nms_bev(
                        boxes_for_nms[label_mask],
                        top_scores[label_mask],
                        thresh=nms_thr,
                        pre_max_size=pre_max_size[i],
                        post_max_size=post_max_size[i])
                    indices = all_indices[label_mask][selected]
                    all_selected_mask.scatter_(0, indices, True)
            else:
                all_selected_mask = []

            # if selected is not None:
            selected_boxes = box_preds[all_selected_mask]
            selected_labels = top_labels[all_selected_mask]
            selected_scores = top_scores[all_selected_mask]

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                predictions_dict = dict(
                    bboxes=final_box_preds,
                    scores=final_scores,
                    labels=final_labels)
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size],
                                       dtype=dtype,
                                       device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0],
                                       dtype=top_labels.dtype,
                                       device=device))

            predictions_dicts.append(predictions_dict)
        return predictions_dicts


    def _decode_all_preds(self,
                          pred_dict,
                          point_cloud_range=None,
                          voxel_size=None):
        batch_size, _, H, W = pred_dict['reg'].shape

        batch_center = pred_dict['reg'].permute(0, 2, 3, 1).contiguous().view(
            batch_size, H * W, 2)  # (B, H, W, 2)
        batch_center_z = pred_dict['height'].permute(
            0, 2, 3, 1).contiguous().view(batch_size, H * W, 1)  # (B, H, W, 1)
        batch_dim = pred_dict['dim'].exp().permute(
            0, 2, 3, 1).contiguous().view(batch_size, H * W, 3)  # (B, H, W, 3)
        batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1).permute(
            0, 2, 3, 1).contiguous().view(batch_size, H * W, 1)  # (B, H, W, 1)
        batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1).permute(
            0, 2, 3, 1).contiguous().view(batch_size, H * W, 1)  # (B, H, W, 1)
        batch_vel = pred_dict['vel'].permute(0, 2, 3, 1).contiguous().view(
            batch_size, H * W, 2) if 'vel' in pred_dict.keys() else None

        angle = torch.atan2(batch_rot_sin, batch_rot_cos)  # (B, H*W, 1)

        ys, xs = torch.meshgrid([
            torch.arange(
                0, H, device=batch_center.device, dtype=batch_center.dtype),
            torch.arange(
                0, W, device=batch_center.device, dtype=batch_center.dtype)
        ])
        ys = ys.view(1, H, W).repeat(batch_size, 1, 1)
        xs = xs.view(1, H, W).repeat(batch_size, 1, 1)
        xs = xs.view(batch_size, -1, 1) + batch_center[:, :, 0:1]
        ys = ys.view(batch_size, -1, 1) + batch_center[:, :, 1:2]

        xs = xs * voxel_size[0] + point_cloud_range[0]
        ys = ys * voxel_size[1] + point_cloud_range[1]

        box_part_list = [xs, ys, batch_center_z, batch_dim, angle]
        if batch_vel is not None:
            box_part_list.append(batch_vel)

        box_preds = torch.cat((box_part_list),
                              dim=-1).view(batch_size, H, W, -1)

        return box_preds
    
    def calc_iou_loss(self, iou_preds, batch_box_preds, mask, ind, gt_boxes):
        """
        Args:
            iou_preds: (batch x 1 x h x w)
            batch_box_preds: (batch x (7 or 9) x h x w)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            gt_boxes: List of batch groundtruth boxes.

        Returns:
            Tensor: IoU Loss.
        """
        if mask.sum() == 0:
            return iou_preds.new_zeros((1))

        mask = mask.bool()
        selected_iou_preds = self._transpose_and_gather_feat(iou_preds,
                                                             ind)[mask]

        selected_box_preds = self._transpose_and_gather_feat(
            batch_box_preds, ind)[mask]
        gt_boxes = torch.cat(gt_boxes)
        assert gt_boxes.size(0) == selected_box_preds.size(0)
        iou_target = boxes_iou3d(selected_box_preds[:, 0:7], gt_boxes[:, 0:7])
        iou_target = torch.diag(iou_target).view(-1)
        iou_target = iou_target * 2 - 1  # [0, 1] ==> [-1, 1]

        loss = self.loss_iou(selected_iou_preds.view(-1), iou_target)
        loss = loss / torch.clamp(mask.sum(), min=1e-4)
        return loss
    
    def _transpose_and_gather_feat(self, feat, ind):
            feat = feat.permute(0, 2, 3, 1).contiguous()
            feat = feat.view(feat.size(0), -1, feat.size(3))
            feat = self._gather_feat(feat, ind)
            return feat
        
    def calc_iou_reg_loss(self, batch_box_preds, mask, ind, gt_boxes):
        if mask.sum() == 0:
            return batch_box_preds.new_zeros((1))

        mask = mask.bool()

        selected_box_preds = self._transpose_and_gather_feat(
            batch_box_preds, ind)[mask]
        gt_boxes = torch.cat(gt_boxes)
        assert gt_boxes.size(0) == selected_box_preds.size(0)
        loss = self.loss_iou_reg(selected_box_preds[:, 0:7], gt_boxes[:, 0:7])

        return loss