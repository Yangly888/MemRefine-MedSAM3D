"""Utility functions for training and evaluation.
    Yunli Qi
"""

import logging
import os
import random
import sys
import time
from datetime import datetime

import dateutil.tz
import numpy as np
import torch
from torch.autograd import Function

import cfg
from sam2_train.modeling.cpgf import CPGF

args = cfg.parse_args()
device = torch.device('cuda', args.gpu_device)

def get_network(args, net, use_gpu=True, gpu_device = 0, distribution = True):
    """ return given network
    """

    if net == 'sam2':
        from sam2_train.build_sam import build_sam2_video_predictor

        sam2_checkpoint = args.sam_ckpt
        model_cfg = args.sam_config

        net = build_sam2_video_predictor(config_file=model_cfg, ckpt_path=sam2_checkpoint, mode=None)
        
        # Load pretrained weights if specified
        if args.weights != 0 and args.weights is not None:
            if os.path.exists(args.weights):
                print(f"Loading pretrained weights from: {args.weights}")
                try:
                    checkpoint = torch.load(args.weights, map_location='cpu')
                    # Handle different checkpoint formats
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # Load state dict with strict=False to handle missing/extra keys
                    missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)
                    if missing_keys:
                        print(f"Missing keys: {len(missing_keys)} keys")
                    if unexpected_keys:
                        print(f"Unexpected keys: {len(unexpected_keys)} keys")
                    print("Pretrained weights loaded successfully!")
                except Exception as e:
                    print(f"Error loading weights: {e}")
                    print("Continuing with default SAM2 initialization...")
            else:
                print(f"Warning: Weight file not found at {args.weights}")
                print("Continuing with default SAM2 initialization...")
        
        # Pass use_lgff parameter to the model
        if hasattr(net.sam_mask_decoder, 'use_lgff'):
            net.sam_mask_decoder.use_lgff = args.use_lgff
            print(f"LGFF module {'enabled' if args.use_lgff else 'disabled'}")
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        net = net.to(device=gpu_device)

    # === CPGF wiring (3D) ===
    if getattr(args, 'cpgf', 0):
        feat_dim = getattr(net, 'feat_dim', 256)
        net.cpgf = CPGF(C=feat_dim, heads=args.cpgf_heads, dim_head=args.cpgf_dim, dim=3, alpha_cap=args.cpgf_alpha_cap)
        net.cpgf = net.cpgf.to(device=gpu_device)  # 只设置device，不要强制dtype（保持与src一致）
        net.cpgf_topk = args.cpgf_topk

        # 从 memory_bank / inference_state / 缓存中收集 top-k 记忆
        def _collect_mem_feats_3d(self, topk=None):
            # 确保始终使用命令行设置的topk值
            if topk is None:
                topk = getattr(self, 'cpgf_topk', 6)
            mem_feats = []
            if hasattr(self, 'memory_bank') and self.memory_bank is not None:
                if hasattr(self.memory_bank, 'topk_values'):
                    try:
                        # 利用现有 memory bank 的 top-K 记忆
                        topk_features = self.memory_bank.topk_values[:topk]
                        for feat in topk_features:
                            if feat is not None and len(feat.shape) >= 3:
                                mem_feats.append(feat)
                    except Exception as e:
                        print(f"Memory bank access error: {e}")

                # 如果 memory_bank 有 frame features，也可以尝试获取
                if hasattr(self.memory_bank, 'frames') and len(mem_feats) < topk:
                    try:
                        available_frames = list(self.memory_bank.frames.keys())
                        for frame_id in available_frames[-topk:]:
                            frame_data = self.memory_bank.frames[frame_id]
                            if hasattr(frame_data, 'curr_enc_features_0'):
                                mem_feats.append(frame_data.curr_enc_features_0)
                    except Exception as e:
                        print(f"Frame memory access error: {e}")

            # 如果仍然没有足够记忆，使用推理状态中的特征
            if len(mem_feats) < topk and hasattr(self, 'inference_state'):
                try:
                    if hasattr(self.inference_state, 'cached_features'):
                        cached_feats = list(self.inference_state.cached_features.values())
                        for feat in cached_feats[-topk:]:
                            if feat is not None:
                                mem_feats.append(feat)
                except Exception as e:
                    print(f"Inference state memory access error: {e}")

            # 最后回退：如果还是没有足够记忆，构造z±1邻近切片作为伪记忆
            if len(mem_feats) < max(2, topk):  # 至少确保2个记忆
                try:
                    # 获取当前输入特征（如果可用）
                    current_input = getattr(self, '_current_input', None)
                    if current_input is not None and len(current_input.shape) >= 4:
                        B, C, D, H, W = current_input.shape if len(current_input.shape) == 5 else (current_input.shape[0], current_input.shape[1], 1, current_input.shape[2], current_input.shape[3])

                        # 创建z±1邻近切片作为伪记忆
                        needed = max(2, topk) - len(mem_feats)
                        for offset in [1, -1, 2, -2, 3, -3][:needed]:
                            if D > abs(offset):
                                if len(current_input.shape) == 5:
                                    # 5D张量: (B, C, D, H, W)
                                    if D + offset >= 0 and D + offset < D:
                                        shifted_feat = torch.roll(current_input, shifts=offset, dims=2)
                                        mem_feats.append(shifted_feat.detach())
                                else:
                                    # 4D张量: (B, C, H, W) - 在空间维度上偏移
                                    shifted_feat = torch.roll(current_input, shifts=(offset, offset), dims=(-2, -1))
                                    mem_feats.append(shifted_feat.detach())

                                if len(mem_feats) >= max(2, topk):
                                    break
                            elif len(mem_feats) < 2:
                                # 如果D维不足，尝试其他变换作为兜底
                                if current_input is not None:
                                    # 添加噪声作为变体
                                    noise_factor = 0.01 * abs(offset)
                                    noisy_feat = current_input + torch.randn_like(current_input) * noise_factor
                                    mem_feats.append(noisy_feat.detach())
                                if len(mem_feats) >= 2:
                                    break
                except Exception as e:
                    print(f"Neighbor slice fallback error: {e}")

            # 质量筛选：移除异常的记忆特征
            valid_mem_feats = []
            for feat in mem_feats:
                if feat is not None and feat.numel() > 0 and not torch.isnan(feat).any() and not torch.isinf(feat).any():
                    valid_mem_feats.append(feat)

            # 确保至少返回2个有效记忆，如果不足就返回空列表（让调用方跳过CPGF）
            final_mem_feats = valid_mem_feats[:topk] if len(valid_mem_feats) >= 2 else []

            return final_mem_feats

        # 注册方法到网络
        import types
        net._collect_mem_feats_3d = types.MethodType(_collect_mem_feats_3d, net)

        # 注册 pre-hook 用于 CPGF-Pre 残差增强
        import weakref
        from torch.utils.checkpoint import checkpoint

        def _sam_md_pre_hook(mod, inputs):
            if not isinstance(inputs, tuple) or len(inputs) == 0:
                return inputs
            x = inputs[0]
            net_ref = getattr(mod, '_outer_ref', None)
            if net_ref is None:
                return inputs
            net_i = net_ref()
            if net_i is None:
                return inputs

            # LGFF开启时禁用pre-hook，避免前后双注入互相打架
            if hasattr(net_i, 'lgff') and net_i.lgff is not None:
                return inputs

            # 确认通道匹配（不再重建 CPGF；只要第一次创建时C对了即可）
            C_in = x.shape[1]
            assert getattr(net_i.cpgf, 'in_channels', C_in) == C_in, \
                f'CPGF in_channels={getattr(net_i.cpgf,"in_channels",None)} != feature C={C_in}'

            # 拿到 top-K 记忆并全部 detach，确保不回传梯度 & 省图
            mem = [m.detach().to(x.device).to(x.dtype) for m in net_i._collect_mem_feats_3d()]  # 函数内部会使用self.cpgf_topk

            # 如果没有记忆特征，使用EMA稳定记忆以保证CPGF能正常训练
            if len(mem) == 0:
                # 使用EMA维护稳定的单一记忆
                if not hasattr(net_i, '_mem_cache'):
                    # 初始化：直接使用当前特征
                    net_i._mem_cache = x.detach()
                else:
                    # EMA更新：0.9*历史 + 0.1*当前，减少抖动
                    # 确保EMA缓存与当前输入的设备和类型对齐
                    net_i._mem_cache = net_i._mem_cache.to(x.device).to(x.dtype)
                    net_i._mem_cache = 0.9 * net_i._mem_cache + 0.1 * x.detach()

                # 返回单一稳定记忆，确保设备和类型对齐
                mem = [net_i._mem_cache.to(x.device).to(x.dtype)]

            # CPGF与LGFF协同：CPGF在LGFF后做记忆增强，完全解耦
            if len(mem) > 0:
                def run(xi):
                    # dtype对齐保护
                    target_dtype = xi.dtype
                    if hasattr(net_i.cpgf, 'to'):
                        net_i.cpgf.to(dtype=target_dtype)

                    # CPGF输入x（LGFF输出特征），mem_feats（memory bank记忆特征）
                    # 核心：MH-Attn(x, concat(mem_feats)) + 自适应门控α
                    # 输出：x + α·Δ（残差记忆注入）
                    return net_i.cpgf(xi, mem)
                x_enhanced = checkpoint(run, x)
                return (x_enhanced, *inputs[1:])

            return (x, *inputs[1:])

        net.sam_mask_decoder._outer_ref = weakref.ref(net)
        net.sam_mask_decoder.register_forward_pre_hook(_sam_md_pre_hook)

        # 为解码器的CPGF-Post也设置外部网络引用（用于记忆收集）
        # 这样解码器可以调用 net._collect_mem_feats_3d() 和 net.cpgf
        print("已为解码器设置外部网络引用，支持CPGF-Post调用")
    # === /CPGF wiring (3D) ===

    return net

def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger

def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))

def random_click(mask, point_labels = 1, seed=None):
    # check if all masks are black
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        point_labels = max_label
    # max agreement position
    indices = np.argwhere(mask == max_label) 
    # return point_labels, indices[np.random.randint(len(indices))]
    if seed is not None:
        rand_instance = random.Random(seed)
        rand_num = rand_instance.randint(0, len(indices) - 1)
    else:
        rand_num = random.randint(0, len(indices) - 1)
    output_index_1 = indices[rand_num][0]
    output_index_0 = indices[rand_num][1]
    return point_labels, np.array([output_index_0, output_index_1])

def generate_bbox(mask, variation=0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # check if all masks are black
    if len(mask.shape) != 2:
        current_shape = mask.shape
        raise ValueError(f"Mask shape is not 2D, but {current_shape}")
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan])
    # max agreement position
    indices = np.argwhere(mask == max_label) 
    # return point_labels, indices[np.random.randint(len(indices))]
    # print(indices)
    x0 = np.min(indices[:, 0])
    x1 = np.max(indices[:, 0])
    y0 = np.min(indices[:, 1])
    y1 = np.max(indices[:, 1])
    w = x1 - x0
    h = y1 - y0
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2
    if variation > 0:
        num_rand = np.random.randn() * variation
        w *= 1 + num_rand[0]
        h *= 1 + num_rand[1]
        x1 = mid_x + w / 2
        x0 = mid_x - w / 2
        y1 = mid_y + h / 2
        y0 = mid_y - h / 2
    return np.array([y0, x0, y1, x1])

def eval_seg(pred,true_mask_p,threshold):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
            cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p [:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            iou_d += iou(disc_pred,disc_mask)
            iou_c += iou(cup_pred,cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()
            
        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    elif c > 2: # for multi-class segmentation > 2 classes
        ious = [0] * c
        dices = [0] * c
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            for i in range(0, c):
                pred = vpred_cpu[:,i,:,:].numpy().astype('int32')
                mask = gt_vmask_p[:,i,:,:].squeeze(1).cpu().numpy().astype('int32')
        
                '''iou for numpy'''
                ious[i] += iou(pred,mask)

                '''dice for torch'''
                dices[i] += dice_coeff(vpred[:,i,:,:], gt_vmask_p[:,i,:,:]).item()
            
        return tuple(np.array(ious + dices) / len(threshold)) # tuple has a total number of c * 2
    else:
        eiou, edice = 0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            eiou += iou(disc_pred,disc_mask)

            '''dice for torch'''
            edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            
        return eiou / len(threshold), edice / len(threshold)
    
def iou(outputs: np.array, labels: np.array):

    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)


    return iou.mean()

def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target
