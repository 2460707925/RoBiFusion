import argparse
import mmcv
import os
import torch
import torch.nn.functional as F
import warnings
from types import MethodType
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.core import bbox3d2result
# def add_gt_depth_feat_hook(self,module,input,output):
from mmdet3d.models.bi_fusion_layers.depth_refinement import DepthRefinementModule
depth_refine_module=DepthRefinementModule()
    
saved_bev_feats_src=[]
saved_bev_feats_tgt=[]

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--debug",action="store_true",help="debug mode")
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def forward_test(self, points=None, img_metas=None, img=None, **kwargs):
    """
    Args:
        points (list[torch.Tensor]): the outer list indicates test-time
            augmentations and inner torch.Tensor should have a shape NxC,
            which contains all points in the batch.
        img_metas (list[list[dict]]): the outer list indicates test-time
            augs (multiscale, flip, etc.) and the inner list indicates
            images in a batch
        img (list[torch.Tensor], optional): the outer
            list indicates test-time augmentations and inner
            torch.Tensor should have a shape NxCxHxW, which contains
            all images in the batch. Defaults to None.
    """
    """Test function without augmentaiton."""
    global depth_refine_module,saved_bev_feats_src,saved_bev_feats_src,saved_bev_feats_tgt
    pts_feats,depth_feats,context_feats,pts_coors = self.extract_feats(points, img, img_metas)
    src_bev_feats,_,_=self.extract_bev_feats(pts_feats,depth_feats,context_feats ,pts_coors ,img_metas)
    gt_depth_feats_max,gt_depth_feats_min=generate_gt_depth_feat(kwargs['gt_depth'],self.loss_depth)
    gt_depth_feats=F.normalize(gt_depth_feats_max+gt_depth_feats_min,dim=1,p=1)
    fuse_depth_feats=depth_refine_module(depth_feats,gt_depth_feats)
    tgt_bev_feats,_,_=self.extract_bev_feats(pts_feats,fuse_depth_feats,context_feats ,pts_coors ,img_metas)
    if len(saved_bev_feats_src)<=10:
        saved_bev_feats_src.append(src_bev_feats)
        saved_bev_feats_tgt.append(tgt_bev_feats)
    outs = self.bev_bbox_head(tgt_bev_feats)
    bev_bbox_list = self.bev_bbox_head.get_bboxes(outs, img_metas, rescale=False)
    bbox_results = [bbox3d2result(bboxes, scores, labels)
                    for bboxes, scores, labels in bev_bbox_list
                ]
    bbox_list = [dict() for i in range(len(img_metas))]
    for result_dict, bev_bbox in zip(bbox_list, bbox_results):
        result_dict['pts_bbox'] = bev_bbox
    return bbox_list
    
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
    gt_depths_max=torch.max(gt_depths,dim=-1).values
    
    gt_depths_tmp = torch.where(gt_depths == 0.0,
                                1e5 * torch.ones_like(gt_depths),
                                gt_depths)
    gt_depths_min = torch.min(gt_depths_tmp, dim=-1).values
    
    gt_depths_max=gt_depths_max.view(B * N, H // self.downsample_factor, W // self.downsample_factor)
    gt_depths_min = gt_depths_min.view(B * N, H // self.downsample_factor, W // self.downsample_factor)
    
    gt_depths_max = (gt_depths_max - (self.dbound[0] - self.dbound[2])) / self.dbound[2]
    gt_depths_min = (gt_depths_min - (self.dbound[0] - self.dbound[2])) / self.dbound[2]
    
    gt_depths_max = torch.where(
        (gt_depths_max < self.depth_channels + 1) & (gt_depths_max >= 0.0),
        gt_depths_max, torch.zeros_like(gt_depths_max))

    gt_depths_min = torch.where(
        (gt_depths_min < self.depth_channels + 1) & (gt_depths_min >= 0.0),
        gt_depths_min, torch.zeros_like(gt_depths_min))

    gt_depths_max = F.one_hot(gt_depths_max.long(),num_classes=self.depth_channels + 1).view(-1, self.depth_channels + 1)[:, 1:]
    gt_depths_min = F.one_hot(gt_depths_min.long(),num_classes=self.depth_channels + 1).view(-1, self.depth_channels + 1)[:, 1:]

    return gt_depths_max.float(),gt_depths_min.float()
    
def generate_gt_depth_feat(gt_depths, depth_module):
    B, N, H, W = gt_depths.shape
    h,w=H // depth_module.downsample_factor,W // depth_module.downsample_factor     
    depth_labels_max,depth_labels_min=get_downsampled_gt_depth(depth_module,gt_depths)
    depth_labels_max=depth_labels_max.reshape(B,N,h,w,-1)
    depth_labels_min=depth_labels_min.reshape(B,N,h,w,-1)
    
    output_shape=depth_labels_max.shape
    
    depth_labels_max=torch.argmax(depth_labels_max,axis=-1)
    depth_labels_min=torch.argmax(depth_labels_min,axis=-1)
    
    depth_feature_max=generate_depth_feature(output_shape,depth_labels_max).view(B*N,h,w,-1).permute(0,3,1,2)
    depth_feature_min=generate_depth_feature(output_shape,depth_labels_min).view(B*N,h,w,-1).permute(0,3,1,2)
    return depth_feature_max,depth_feature_min
    
def generate_depth_feature(output_shape,depth_index):
    device=depth_index.device
    B,N,h,w,depth_dim=output_shape
    sample_range=[-6,6]
    d_coors1=torch.linspace(sample_range[0],0,depth_dim)
    d_coors2=torch.linspace(0,sample_range[1],depth_dim)
    d_coors=torch.cat((d_coors1,d_coors2))
    mask = torch.arange(len(d_coors)).unsqueeze(0)
    mask = mask.expand(B,N,h,w,-1).to(device=device)
    mask=(mask>=(depth_dim-1-depth_index).unsqueeze(-1)) & (mask<=(2*depth_dim-2-depth_index).unsqueeze(-1))
    x=d_coors.expand(B,N,h,w,-1)[mask].view(B,N,h,w,-1).to(device=device)
    pts_conf=torch.ones(x.shape[:-1]).unsqueeze(-1).to(device=device)*10
    dist=torch.distributions.Normal(loc=0, scale=pts_conf)
    pts_depth_probs=F.normalize(torch.exp(dist.log_prob(x)),p=2,dim=-1)
    return pts_depth_probs


def main():
    args = parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(5620)
        print("Wait for debug")
        debugpy.wait_for_client()
        print("Start debug")


    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model.forward_test=MethodType(forward_test,model)
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))
            
    torch.save(saved_bev_feats_src,"visualize/bev_feats1/saved_bev_feats_src.pth")
    torch.save(saved_bev_feats_tgt,"visualize/bev_feats1/saved_bev_feats_tgt.pth")


if __name__ == '__main__':
    main()
