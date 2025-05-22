import warnings
from mmcv.utils import Registry
from mmdet.models.builder import MODELS as MMDET_MODELS
from mmdet.models.builder import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                                  ROI_EXTRACTORS, SHARED_HEADS)
from .registry import FUSION_LAYERS, MIDDLE_ENCODERS, VOXEL_ENCODERS

MODELS = Registry('models', parent=MMDET_MODELS)


VOXEL_ENCODERS = MODELS
MIDDLE_ENCODERS = MODELS
FUSION_LAYERS = MODELS
BI_FUSION_LAYERS=MODELS
IMG_BEV_ENCODER=MODELS
FUSION_BEV_ENCODER=MODELS
DEPTH_NET=MODELS

def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_roi_extractor(cfg):
    """Build RoI feature extractor."""
    return ROI_EXTRACTORS.build(cfg)


def build_shared_head(cfg):
    """Build shared head of detector."""
    return SHARED_HEADS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss function."""
    return LOSSES.build(cfg)


def build_voxel_encoder(cfg):
    """Build voxel encoder."""
    return VOXEL_ENCODERS.build(cfg)


def build_middle_encoder(cfg):
    """Build middle level encoder."""
    return MIDDLE_ENCODERS.build(cfg)


def build_fusion_layer(cfg):
    """Build fusion layer."""
    return FUSION_LAYERS.build(cfg)


def build_bi_fusion_layer(cfg):
    return BI_FUSION_LAYERS.build(cfg)


def build_img_bev_encoder(cfg):
    return IMG_BEV_ENCODER.build(cfg)


def build_fusion_bev_encoder(cfg):
    return FUSION_BEV_ENCODER.build(cfg)

def build_depth_net(cfg):
    return DEPTH_NET.build(cfg)

def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))