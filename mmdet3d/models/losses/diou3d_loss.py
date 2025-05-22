from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from mmdet.models.losses.utils import weighted_loss
from torch import Tensor
from ..builder import LOSSES

def center_to_corner2d(center, dim):
    corners_norm = torch.tensor(
        [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]],
        device=dim.device).type_as(center)  # (4, 2)
    corners = dim.view([-1, 1, 2]) * corners_norm.view([1, 4, 2])  # (N, 4, 2)
    corners = corners + center.view(-1, 1, 2)
    return corners


@weighted_loss
def diou3d_loss(pred_boxes, gt_boxes, eps: float = 1e-7):
    """
    modified from https://github.com/agent-sgs/PillarNet/blob/master/det3d/core/utils/center_utils.py # noqa
    Args:
        pred_boxes (N, 7):
        gt_boxes (N, 7):

    Returns:
        Tensor: Distance-IoU Loss.
    """
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = center_to_corner2d(pred_boxes[:, :2],
                                  pred_boxes[:, 3:5])  # (N, 4, 2)
    gcorners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:,
                                                            3:5])  # (N, 4, 2)

    inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])
    out_max_xy = torch.maximum(qcorners[:, 2], gcorners[:, 2])
    out_min_xy = torch.minimum(qcorners[:, 0], gcorners[:, 0])

    # calculate area
    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(
        pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5],
        gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) - torch.maximum(
            pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5],
            gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter + eps

    # boxes_iou3d_gpu(pred_boxes, gt_boxes)
    inter_diag = torch.pow(gt_boxes[:, 0:3] - pred_boxes[:, 0:3], 2).sum(-1)

    outer_h = torch.maximum(
        gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5],
        pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - torch.minimum(
            gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5],
            pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    outer_h = torch.clamp(outer_h, min=0)
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = outer[:, 0]**2 + outer[:, 1]**2 + outer_h**2 + eps

    dious = volume_inter / volume_union - inter_diag / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)

    loss = 1 - dious

    return loss


@LOSSES.register_module()
class DIoU3DLoss(nn.Module):
    r"""3D bboxes Implementation of `Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression <https://arxiv.org/abs/1911.08287>`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        eps (float): Epsilon to avoid log(0). Defaults to 1e-6.
        reduction (str): Options are "none", "mean" and "sum".
            Defaults to "mean".
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * diou3d_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss