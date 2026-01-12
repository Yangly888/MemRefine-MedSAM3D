# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Yunli Qi
"""

import os
import time
import random
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import cfg
from func_3d import function
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    # 过滤MONAI的警告信息
    warnings.filterwarnings("ignore", message="single channel prediction, `to_onehot_y=True` ignored.")

    args = cfg.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    #net.to(dtype=torch.bfloat16)
    if args.pretrain:
        print(args.pretrain)
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights,strict=False)

    sam_layers = (
                  []
                #   + list(net.image_encoder.parameters())
                #   + list(net.sam_prompt_encoder.parameters())
                  + list(net.sam_mask_decoder.parameters())
                  + (list(net.cpgf.parameters()) if hasattr(net, 'cpgf') else [])
                  )
    mem_layers = (
                  []
                  + list(net.obj_ptr_proj.parameters())
                  + list(net.memory_encoder.parameters())
                  + list(net.memory_attention.parameters())
                  + list(net.mask_downsample.parameters())
                 
                  )
    if len(sam_layers) == 0:
        optimizer1 = None
        scheduler1 = None
    else:
        optimizer1 = optim.Adam(sam_layers, lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        # SAM层（包括CPGF）保持学习率不变
        scheduler1 = None
    if len(mem_layers) == 0:
        optimizer2 = None
        scheduler2 = None

    #torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    nice_train_loader, nice_test_loader = get_dataloader(args)

    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    '''begain training'''
    best_acc = 0.0
    best_tol = 1e4
    best_dice = 0.0

    for epoch in range(settings.EPOCH):

        # if epoch < 5:
        #     tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
        #     logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
        net.train()
        # 让模型知道当前epoch（供CPGF门控暖启动使用）
        if hasattr(net, 'set_current_epoch'):
            net.set_current_epoch(epoch)
        time_start = time.time()
        loss, prompt_loss, non_prompt_loss = function.train_sam(args, net, optimizer1, optimizer2, nice_train_loader, epoch)
        logger.info(f'Train loss: {loss}, {prompt_loss}, {non_prompt_loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        net.eval()
        if epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
            tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)

            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

            # 保存最佳模型
            if edice > best_dice:
                best_dice = edice
                best_acc = eiou
                best_tol = tol
                torch.save({'model': net.state_dict(), 'epoch': epoch, 'dice': edice, 'iou': eiou},
                          os.path.join(args.path_helper['ckpt_path'], 'best_model.pth'))
                logger.info(f'New best model saved! DICE: {edice:.4f}, IOU: {eiou:.4f}')

            torch.save({'model': net.state_dict()}, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))

        # 手动学习率调整策略
        # SAM层（包括CPGF）：始终保持1e-4不变
        # 记忆层：前70轮1e-8，之后提升到1e-5以增强记忆学习（参考CPGF-only在后期的优势）
        if optimizer2 is not None and epoch == 70:
            for param_group in optimizer2.param_groups:
                param_group['lr'] = 1e-5
            logger.info(f'Memory layer learning rate increased to 1e-5 at epoch {epoch}')

        # 在训练后期进一步提升记忆层学习率，以充分发挥CPGF+LGFF协同优势
        if optimizer2 is not None and epoch == 85:
            for param_group in optimizer2.param_groups:
                param_group['lr'] = 2e-5
            logger.info(f'Memory layer learning rate further increased to 2e-5 at epoch {epoch}')

        # 记录当前学习率
        if epoch % 10 == 0:
            current_lr1 = optimizer1.param_groups[0]['lr'] if optimizer1 is not None else 0
            current_lr2 = optimizer2.param_groups[0]['lr'] if optimizer2 is not None else 0
            logger.info(f'Learning rates at epoch {epoch}: SAM={current_lr1:.2e}, Memory={current_lr2:.2e}')

    writer.close()


if __name__ == '__main__':
    main()