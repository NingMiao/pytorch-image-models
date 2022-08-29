#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torch.optim as optim

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, \
    LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, convert_splitbn_model, convert_sync_batchnorm, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler


torch.backends.cudnn.benchmark = True

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside of the dataset group because it is positional.
parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset')
group.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
group.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
group.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
group.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
group.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
group.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (Model default if None)')
group.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
group.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
group.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
group.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset')
group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='Validation batch size override (default: None)')
group.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='torch.jit.script the full model')
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                    help="Enable AOT Autograd support. (It's recommended to use this option with `--fuser nvfuser` together)")
group.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=2e-5,
                    help='weight decay (default: 2e-5)')
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
group.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')
group.add_argument('--layer-decay', type=float, default=None,
                    help='layer-wise learning rate decay (default: None)')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
group.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.05)')
group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
group.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
group.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
group.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
group.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
group.add_argument('--decay-milestones', default=[30, 60], type=int, nargs='+', metavar="MILESTONES",
                    help='list of decay epoch indices for multistep lr. must be increasing')
group.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
group.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
group.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
group.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
group.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
group.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
group.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
group.add_argument('--aug-repeats', type=float, default=0,
                    help='Number of augmentation repetitions (distributed training only) (default: 0)')
group.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
group.add_argument('--jsd-loss', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
group.add_argument('--bce-loss', action='store_true', default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')
group.add_argument('--bce-target-thresh', type=float, default=None,
                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
group.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
group.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
group.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
group.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
group.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
group.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
group.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
group.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
group.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
group.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
group.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
group.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
group.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
group.add_argument('--worker-seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')

group.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
group.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
group.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
group.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')

# Device
group = parser.add_argument_group('Device')
group.add_argument('--device', type=str, default='cpu',
                    help='cpu/cuda/tpu-1/tp')

# Learnable Invariance
group = parser.add_argument_group('Li')
group.add_argument('-Li_config_path', type=str, default='', help='')
group.add_argument('-target_entropy', type=float, default=3.0, help='log(17)=2.83')
group.add_argument('-resume_Li', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('-Li_save_path', type=str, default='', help='')
group.add_argument('-entropy_weights', type=float, default=0.0, help='')

group.add_argument('-eval_only', action='store_true', default=False,
                    help='') 
group.add_argument('-save_every', type=int, default=10,
                    help='') 
                   
def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    
    if args.Li_config_path:
        Li_configs=yaml.safe_load(open(args.Li_config_path,'r'))
        if args.entropy_weights:
            Li_configs['entropy_weights']=args.entropy_weights  
        import sys
        sys.path.insert(0, '../InstaAug/')
        global learnable_invariance
        from InstaAug_module import learnable_invariance
        
    else:
        Li_configs={'li_flag': False}
    
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text, Li_configs


def main():
    utils.setup_default_logging()
    args, args_text, Li_configs = _parse_args()
    
    if args.device.startswith('tpu'):
        args.tpu_core_num=int(args.device[4:])
        args.batch_size=int(args.batch_size//args.tpu_core_num)
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as met
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch_xla.utils.utils as xu
        global xm, xmp, pl
        torch.set_default_tensor_type('torch.FloatTensor')
        
        def log_fn(string):
            xm.master_print(string)
            
        
        xmp.spawn(run_wrapper, args=(args, args_text, log_fn, Li_configs), nprocs=args.tpu_core_num, start_method='fork')
    else:
        run_wrapper('', args, args_text, print, Li_configs)
    
def reduce_fn(vals):
    return sum(vals) / len(vals)
    
def run_wrapper(_, args, args_text, log_fn, Li_configs):
    
    if args.log_wandb and xm.is_master_ordinal():
        wandb.init(project=args.experiment, config=args)

    torch.manual_seed(124)
    
    #Initialize model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint)
    
    if args.device.startswith('tpu'):
        device = xm.xla_device()
        model = xmp.MpModelWrapper(model)
        model=model.to(device)
    else:
        device=args.device

    log_fn(f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')
    
    #Initialize Li net
    if Li_configs['li_flag']:
        Li=learnable_invariance(Li_configs, device=device)  
        if args.device.startswith('tpu'):
            Li = xmp.MpModelWrapper(Li)
            Li=Li.to(device)
            log_fn(f'Li net created, param count:{sum([m.numel() for m in Li.parameters()])}')
    else:
        Li=None
        log_fn('No Li net')
    
    
    
    data_config = resolve_data_config(vars(args), model=model)

    # setup augmentation batch splits for contrastive loss
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits
    
    #Create optimizer
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    if Li_configs['li_flag']:
        optimizer_Li=optim.SGD(Li.parameters(), lr=Li_configs['lr'])
    
    
    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        model.to('cpu')
        resume_epoch=0
        saved_dict=torch.load(args.resume)
        if 'epoch' in saved_dict:
            resume_epoch=saved_dict['epoch']            
        model.load_state_dict(saved_dict['model'])
        if 'optimizer' in saved_dict and not args.no_resume_opt:
            optimizer.load_state_dict(saved_dict['optimizer'])
        model.to(device)
        log_fn('resume model finished! Epoch: {}'.format(resume_epoch))
    if args.resume_Li and Li_configs['li_flag']:
        Li.to('cpu')
        saved_dict=torch.load(args.resume_Li)
        Li.load_state_dict(saved_dict)
        Li.to(device)
        log_fn('resume Li finished! Epoch: {}'.format(resume_epoch))



    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    log_fn('Scheduled epochs: {}'.format(num_epochs))
    
    if Li_configs['li_flag']:
        lr_scheduler_Li, _ = create_scheduler(args, optimizer_Li)
        if lr_scheduler_Li is not None and start_epoch > 0:
            lr_scheduler_Li.step(start_epoch)
        Li.optimizer=optimizer_Li
        Li.scheduler=lr_scheduler_Li
        Li.target_entropy=args.target_entropy
        Li.start_epoch=start_epoch
    
    
    # create the train and eval datasets
    dataset_train = create_dataset(
        args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        repeats=args.epoch_repeats)
    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size)

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    
    if args.device.startswith('tpu'):
        num_replicas=args.tpu_core_num
        rank=xm.get_ordinal()
    else:
        num_replicas=0
        rank=0
    
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=False,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
        num_replicas=num_replicas,
        rank=rank,
    )
    
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
        num_replicas=num_replicas,
        rank=rank,
    )

    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    
    validate_loss_fn = nn.CrossEntropyLoss()
    if args.device=='cuda':
        train_loss_fn = train_loss_fn.cuda()
        validate_loss_fn = validate_loss_fn.cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.device in ['cpu', 'cuda'] or xm.is_master_ordinal():
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:
        eval_acc_old=0.0
        for epoch in range(start_epoch, num_epochs):
            if not args.eval_only:
                train_metrics = train_one_epoch(
                    epoch, model, loader_train, optimizer, train_loss_fn, args,
                    lr_scheduler=lr_scheduler, output_dir=output_dir, loss_scaler=None, mixup_fn=mixup_fn, device=device, Li=Li, Li_configs=Li_configs)
            else:
                train_metrics={'loss':0.0}
            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, device=device, Li=Li, Li_configs=Li_configs)
            
            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])
            
            if args.device.startswith('tpu'):
                train_metrics['loss']=xm.mesh_reduce('train_loss', train_metrics['loss'], reduce_fn)
                for key in eval_metrics:
                    eval_metrics[key]=xm.mesh_reduce('eval'+key, eval_metrics[key], reduce_fn)
                
            
            if output_dir is not None and (not args.device.startswith('tpu') or xm.is_master_ordinal()):
                utils.update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)
                string='epoch:{}, train_loss:{}, eval_loss:{}, eval_top_1:{}'.format(epoch, train_metrics['loss'], eval_metrics['loss'], eval_metrics['top1'])
                log_fn(string)
                log_fn('-------------------------------------')

            if eval_metrics['top1']>eval_acc_old or (epoch-start_epoch)%args.save_every==0:
                eval_acc_old=eval_metrics['top1']
                if not (args.device.startswith('tpu') and not xm.is_master_ordinal()):
                    state_dict={'epoch':epoch, 'model':model.to('cpu').state_dict(), 'optimizer':optimizer.state_dict(), 'acc': eval_metrics['top1']}
                    torch.save(state_dict, os.path.join(output_dir, 'model'+str(epoch)+'.ckpt'))
                    model.to(device)
    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        log_fn('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


        

def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, output_dir=None,
        loss_scaler=None, mixup_fn=None, device=None, log_fn=print, Li=None, Li_configs={}):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    entropy_m = utils.AverageMeter()

    model.train()
    
    if args.device.startswith('tpu'):
        loader = pl.ParallelLoader(loader, [device])
        loader = loader.per_device_loader(device)
    
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if args.device=='cuda':
            input, target = input.cuda(), target.cuda()
        if mixup_fn is not None:
            input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        
        
        if Li_configs['li_flag']:
            optimizer_Li=Li.optimizer
            train_scheduler_Li=Li.scheduler
            input_Li, logprob, entropy_every=Li(input)
                        
            output = model(input_Li)
            loss_predictor = loss_fn(output, target)
            losses_m.update(loss_predictor.item(), input.size(0))
            
            loss_Li_pre=(loss_predictor.detach()*logprob).mean()+loss_predictor.mean()
            entropy=entropy_every.mean(dim=0)
            entropy_m.update(entropy.mean().item(), input.size(0))
            
            if epoch==Li.start_epoch:
                Li.start_entropy=entropy.mean().detach()
                if args.device.startswith('tpu'):
                    Li.start_entropy=xm.mesh_reduce('start_entropy', Li.start_entropy, reduce_fn)
            
            r=min(1, (epoch-Li.start_epoch)/Li_configs['entropy_increase_period'])
            mid_target_entropy=Li.target_entropy*r+Li.start_entropy*(1-r)
            loss=loss_Li_pre+(entropy.mean()-mid_target_entropy)**2*0.3#!#!#!#!
            
            optimizer.zero_grad()
            optimizer_Li.zero_grad()
            
            loss.backward()
            
            if args.clip_grad is not None:
                utils.dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            
            if args.device.startswith('tpu'):
                xm.optimizer_step(optimizer)
                xm.optimizer_step(optimizer_Li)
            else:
                optimizer.step()
                optimizer_Li.step()
            
        else:        
            output = model(input)
            loss = loss_fn(output, target)
            losses_m.update(loss.item(), input.size(0))
            optimizer.zero_grad()
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                utils.dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
                
            if args.device.startswith('tpu'):
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()
        
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            
            string='Master: Train: {} [{:>4d}/{} ({:>3.0f}%)], Loss: {loss.val:#.4g} ({loss.avg:#.3g}), Time: {batch_time.val:.3f}s, {rate:>7.2f}/s, ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s), LR: {lr:.3e}, Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * 1 / batch_time_m.val,
                        rate_avg=input.size(0) * 1 / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m)
            if Li_configs['li_flag']:
                string+=', entropy:{}'.format(entropy_m.avg)
            
            if not args.device.startswith('tpu') or xm.is_master_ordinal():
                log_fn(string)


        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)
        if Li_configs['li_flag'] and Li.scheduler is not None:
            Li.scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, log_suffix='', device=None, log_fn=print, Li=None, Li_configs={}):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    entropy_m = utils.AverageMeter()
    
    model.eval()

    end = time.time()
    
    if args.device.startswith('tpu'):
        loader = pl.ParallelLoader(loader, [device])
        loader = loader.per_device_loader(device)
    
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if args.device=='cuda':
                input, target = input.cuda(), target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            
            
            if Li_configs['li_flag'] and Li_configs['test_time_aug']:
                n_copies=Li_configs['test_copies']
                input_Li, logprob, entropy_every=Li(input, n_copies=n_copies)
                entropy_m.update(entropy_every, entropy_every.shape[0])
                
                output = model(input_Li) 
                if isinstance(output, (tuple, list)):
                    output = output[0]
                    
                bs=input.shape[0]
                logit=F.log_softmax(output)
                logit=logit.reshape([n_copies, bs, -1]).transpose(0,1)
                logprob_new=logprob.reshape([n_copies, bs]).transpose(0,1).unsqueeze(-1)
                output=torch.log(torch.sum(torch.exp(logit)*torch.exp(logprob_new*0.5), dim=1))        
            
            else:
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            reduced_loss = loss.data

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            
    metric=[('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)]
    if Li_configs['li_flag'] and Li_configs['test_time_aug']:
        metric.extend([('entropy', entropy_m.avg)])

    metrics = OrderedDict(metric)

    return metrics


if __name__ == '__main__':
    main()
