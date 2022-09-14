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
import torch.nn.functional as F

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset, DebugDataset
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, \
    LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, convert_splitbn_model, convert_sync_batchnorm, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

import numpy as np

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
group.add_argument('-entropy_parameter', type=float, default=0.3, help='')
group.add_argument('-Li_lr', type=float, default=0.0, help='')
group.add_argument('-Li_loss', type=str, default='crossentropy', help='[crossentropy, accuracy]')
group.add_argument('-mode', type=str, default='1', help='[1, 2]')
group.add_argument('-n_copies', type=int, default=1, help='any int >=1')



group.add_argument('-eval_only', action='store_true', default=False,
                    help='') 
group.add_argument('-save_every', type=int, default=10,
                    help='') 
group.add_argument('-train_Li_only', action='store_true', default=False,
                    help='') 
group.add_argument('-train_sample_amount', type=int, default=0,
                    help='for debug') 
group.add_argument('-train_only', action='store_true', default=False,
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
        Li_configs['lr']=args.Li_lr
        
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
      
    
    # optionally resume from a checkpoint
    
    resume_epoch = None
    if args.resume:
        model.to('cpu')
        saved_dict=torch.load(args.resume)
        model.load_state_dict(saved_dict)
        model.to(device)
        log_fn('resume model finished! Epoch: {}'.format(resume_epoch))
    
    if args.resume_Li and Li_configs['li_flag']:
        Li.to('cpu')
        saved_dict=torch.load(args.resume_Li)
        Li.augmentation.get_param.conv.load_state_dict(saved_dict)
        Li.to(device)
        log_fn('resume Li finished! Epoch: {}'.format(resume_epoch))

    #Create optimizer
    if Li_configs['li_flag']:
        #optimizer_Li=optim.SGD(Li.parameters(), lr=Li_configs['lr'])
        optimizer_Li=optim.Adam(Li.parameters())



    
    if Li_configs['li_flag']:
        Li.optimizer=optimizer_Li
        Li.target_entropy=args.target_entropy
    
    
    # create the datasets
    import numpy as np
    np.random.seed(125)
    #target_np_array=np.load('samples_full/target.npy')
    input_np_array=np.load('output/samples_full/input.npy').reshape(-1, 3,224,224)#?#!
    #input_np_array=np.random.random([256, 3,224,224]).astype(np.float32)#?
    #logit_np_array=np.load('output/samples_full/logit.npy')
    target_logit=np.load('output/samples_full/target_logit.npy')#?
    target_logit=target_logit[:, 196:-1]#?
    
    list_index=np.arange(input_np_array.shape[0])
    np.random.shuffle(list_index)
    train_num=40000
    eval_num=1000
    train_ids=list_index[:train_num]
    eval_ids=list_index[train_num:train_num+eval_num]
    
    log_fn('train_single_best:{}, eval_single_best:{}'.format(target_logit[train_ids].mean(axis=0).max(), target_logit[eval_ids].mean(axis=0).max()))
    
    log_fn('train_best:{}, eval_best:{}'.format(target_logit[train_ids].max(axis=1).mean(), target_logit[eval_ids].max(axis=1).mean()))
    
    input_tensor=torch.tensor(input_np_array[train_ids])
    target_logit_tensor=torch.tensor(target_logit[train_ids])
    dataset=torch.utils.data.TensorDataset(input_tensor, target_logit_tensor)
    
    eval_input_tensor=torch.tensor(input_np_array[eval_ids])
    eval_target_logit_tensor=torch.tensor(target_logit[eval_ids])
    eval_dataset=torch.utils.data.TensorDataset(eval_input_tensor, eval_target_logit_tensor)
    
    if args.device.startswith('tpu'):
        num_replicas=args.tpu_core_num
        rank=xm.get_ordinal()
    else:
        num_replicas=0
        rank=0
    
    loader =  torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, pin_memory=True)
    eval_loader =  torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, pin_memory=True)

    # setup loss function
    loss_fn = nn.CrossEntropyLoss()
    

    # setup checkpoint saver and eval metric tracking
    
    eval_acc_old=0.0
        
    ave_loss_memory=[]
    
    onehot_mat=torch.eye(49, device=device).to(torch.float32)
    
    t=time.time()
    for epoch in range(500):
                
        metrics = train(epoch, None, loader, loss_fn, args, output_dir=None, mixup_fn=None, device=device, Li=Li, Li_configs=Li_configs, ave_loss_memory=ave_loss_memory, onehot_mat=onehot_mat)
        
        if epoch % 1==0:
            if args.device.startswith('tpu'):
                for key in metrics:
                    metrics[key]=xm.mesh_reduce(key, metrics[key], reduce_fn)
            metrics['epoch'] = epoch
            t_now=time.time()
            metrics['time'] = t_now-t
            t=t_now
            log_fn('train  '+str(metrics))
            
            metrics = train(epoch, None, eval_loader, loss_fn, args, output_dir=None, mixup_fn=None, device=device, Li=Li, Li_configs=Li_configs, ave_loss_memory=ave_loss_memory, onehot_mat=onehot_mat, evaluate=True)
            if args.device.startswith('tpu'):
                for key in metrics:
                    metrics[key]=xm.mesh_reduce(key, metrics[key], reduce_fn)
            metrics['epoch'] = epoch
            log_fn('eval  '+str(metrics))
            
            log_fn('---------------------------------------------------------')

def train(
        epoch, model, loader, loss_fn, args, output_dir=None,
         mixup_fn=None, device=None, log_fn=print, Li=None, Li_configs={}, ave_loss_memory=[], onehot_mat=None, evaluate=False):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = False
    batch_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    
    losses_m = utils.AverageMeter()
    losses_all_m = utils.AverageMeter()
    entropy_m = utils.AverageMeter()
    KL_m = utils.AverageMeter()
    acc_m = utils.AverageMeter()
    
    if args.device.startswith('tpu'):
        loader = pl.ParallelLoader(loader, [device])
        loader = loader.per_device_loader(device)
    
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    
    for batch_idx, (input, logit) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if True:
            optimizer_Li=Li.optimizer
            _, _, entropy_every, KL_every=Li(input)#?
            
            logprob=Li.augmentation.distC_crop.logprob
            prob = torch.exp(logprob)
            
            if args.mode=='1':
                rand = torch.empty_like(prob).uniform_()
                samples = (-rand.log()+logprob).topk(k=args.n_copies).indices
                samples_onehot=torch.index_select(onehot_mat, 0, samples.reshape([-1])).reshape([samples.shape[0], samples.shape[1], -1]).sum(1)
                
                loss = -(logit*samples_onehot*logprob).mean()
                loss_record=-(logit*prob).sum(1).mean()
                
            elif args.mode=='2':
                loss=-(logit*prob).sum(1).mean()
                loss_record=loss
            
            entropy=entropy_every.mean(dim=0)
            loss_all=loss+(-entropy)*0.0
            
            losses_m.update(loss_record.item(), input.size(0))
            losses_all_m.update(loss_all.item(), input.size(0))
            entropy_m.update(entropy.mean().item(), input.size(0))
            KL_m.update(KL_every.mean().item(), input.size(0))
            
            if not evaluate:
                optimizer_Li.zero_grad()
            
                loss_all.backward()

                if args.device.startswith('tpu'):
                    xm.optimizer_step(optimizer_Li)
                else:
                    optimizer_Li.step()
                
    metric=[('loss', losses_m.avg), ('loss_all', losses_all_m.avg)]
    if Li_configs['li_flag']:
        metric.extend([('entropy', entropy_m.avg), ('KL', KL_m.avg)])
    return OrderedDict(metric)


if __name__ == '__main__':
    main()
