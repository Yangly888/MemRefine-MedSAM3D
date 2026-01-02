# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from sam2_train.modeling.sam2_utils import LayerNorm2d, MLP
from .lgff import LGFFBlock  # 你原来已添加

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_multimask_outputs + 1,  # 与 mask tokens 数一致
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )

        # ======================== LGFF 模块（你原有的精修模块） ========================
        self.use_lgff = 0  # Default to disabled, will be overridden by args.use_lgff
        print("初始化 LGFFBlock 作为解码器前的最终增强器...")
        self.lgff_final_refiner = LGFFBlock(
            in_channels=self.transformer_dim,
            dim=2
        )

        # ======================== 新增：门控层（CGR-LGFF 的 α） ========================
        # 说明：我们用 IoU token 的特征 (B, transformer_dim) 通过一个小 MLP → 标量 α∈[0,1]
        # 随后做残差融合：src <- src + α * (refined - src)，高置信少修、低置信多修
        # 优化：增加感知CPGF增强的能力，避免覆盖有益的CPGF修改
        self.gate_mlp = nn.Sequential(
            nn.Linear(transformer_dim, iou_head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),  # 增加轻微正则化
            nn.Linear(iou_head_hidden_dim, 1),
            nn.Sigmoid(),
        )
        with torch.no_grad():
            # 调整初期偏置，让门控更适应CPGF增强后的特征
            last_linear = self.gate_mlp[-2]
            if hasattr(last_linear, "bias") and last_linear.bias is not None:
                last_linear.bias.fill_(-1.5)  # sigmoid(-1.5)≈0.18，稍微提高基础门控

        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]                          # (B, C)
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]  # (B, M, C)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        # ======================== 稳定增长互补增强策略 ========================
        original_dtype = src.dtype
        src_original = src.clone()

        # 获取训练epoch
        current_epoch = 0
        if hasattr(self, '_outer_ref') and self._outer_ref is not None:
            net_ref = self._outer_ref()
            if net_ref is not None:
                current_epoch = getattr(net_ref, '_current_epoch', 0)

        # 并行计算CPGF和LGFF增强
        cpgf_enhanced = src.clone()
        lgff_enhanced = src.clone()

        # CPGF增强 - 适度自适应，避免过早衰落
        if hasattr(self, '_outer_ref') and self._outer_ref is not None:
            net_ref = self._outer_ref()
            if net_ref is not None and hasattr(net_ref, 'cpgf') and net_ref.cpgf is not None:
                mem = [m.detach().to(src.device).to(src.dtype) for m in net_ref._collect_mem_feats_3d()]
                if len(mem) > 0:
                    mem_energy = torch.stack([torch.mean(torch.abs(m)) for m in mem])
                    if torch.mean(mem_energy) > 1e-6:
                        net_ref.cpgf.to(dtype=src.dtype)
                        try:
                            cpgf_delta = net_ref.cpgf(src, mem)
                            if not torch.isnan(cpgf_delta).any():
                                # 关键改进：维持早期高性能，避免立即衰落
                                base_alpha = 0.18  # 提升基础强度（从0.15到0.18）

                                # 前10轮稍微增强，避免第2轮后衰落
                                if current_epoch <= 10:
                                    early_boost = 0.02  # 前期额外2%增强
                                    alpha = base_alpha + early_boost
                                else:
                                    alpha = base_alpha

                                cpgf_enhanced = src + alpha * cpgf_delta
                        except:
                            pass

        # LGFF增强
        if self.use_lgff:
            self.lgff_final_refiner.to(src.device)
            use_amp = src.dtype in (torch.float16, torch.bfloat16)

            with torch.cuda.amp.autocast(enabled=use_amp):
                lgff_enhanced = self.lgff_final_refiner(g=src.float(), x=src.float())
            lgff_enhanced = lgff_enhanced.to(dtype=src.dtype)

        # 融合策略：CPGF主导但给LGFF更多空间
        cpgf_weight = 0.65  # 略微降低CPGF权重（从0.7到0.65）
        lgff_weight = 0.35  # 提升LGFF权重（从0.3到0.35）

        final_enhanced = cpgf_weight * cpgf_enhanced + lgff_weight * lgff_enhanced

        # 关键改进：提升融合比例，维持性能增长
        base_fusion_ratio = 0.68  # 提升基础融合比例（从0.6到0.68）

        # 早期稍微保守，中期开始增强
        if current_epoch <= 5:
            fusion_ratio = base_fusion_ratio - 0.03  # 前5轮稍微保守
        elif current_epoch <= 20:
            fusion_ratio = base_fusion_ratio  # 6-20轮基础强度
        else:
            # 20轮后温和增强，避免后期衰落
            late_boost = min(0.05, (current_epoch - 20) * 0.001)
            fusion_ratio = base_fusion_ratio + late_boost

        fusion_ratio = min(fusion_ratio, 0.75)  # 上限75%，避免过度增强
        src = fusion_ratio * final_enhanced + (1 - fusion_ratio) * src_original
        # ======================== 互补增强策略结束 ========================

        # 这里开始改：
        use_amp = original_dtype in (torch.float16, torch.bfloat16)
        with torch.amp.autocast('cuda', enabled=use_amp):
            if not self.use_high_res_features:
                upscaled_embedding = self.output_upscaling(src)
            else:
                dc1, ln1, act1, dc2, act2 = self.output_upscaling
                feat_s0, feat_s1 = high_res_features
                upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
                upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

            hyper_in_list: List[torch.Tensor] = []
            for i in range(self.num_mask_tokens):
                hyper_in_list.append(
                    self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                )
            hyper_in = torch.stack(hyper_in_list, dim=1)
            b, c, h, w = upscaled_embedding.shape
            masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        # 这里结束改

        # Generate mask quality predictions（保持与你原逻辑一致）
        iou_pred = self.iou_prediction_head(iou_token_out.float())

        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits


    def _get_stability_scores(self, mask_logits):
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        #（原逻辑保持不变）
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
