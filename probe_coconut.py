# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

# [Fix] 适配新版 transformers 的 KV Cache
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None

# [NEW] 在 Outputs 中增加 probes 字段用于监控
Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits", "probes"])
MAX_N_LATENT = 8


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        decoupling_mode="original",  # 模式开关: 'original', 'residual', 'normalized'
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.decoupling_mode = decoupling_mode

        # [NEW] 稀疏性惩罚系数 (L1 Loss Weight)
        # 强迫模型只在少数关键步骤使用高 Alpha (实现二值化/降维效果)
        self.sparsity_weight = 0.002/20

        # [NEW] Normalized 模式的基准缩放因子

        # self.norm_scale_factor = nn.Parameter(torch.tensor([80.0]))

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

        # [NEW] 初始化 Intensity Predictor (MLP)
        if self.decoupling_mode in ["residual", "normalized"]:
            hidden_size = self.embedding.weight.shape[1]

            # 1. 定义 MLP
            self.scale_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1)
            )

            # 2. [关键] 最后一层零初始化 (Zero-Init)
            # 确保初始状态下 alpha/gate 输出严格为 0
            # 这样训练开始时，Residual 模式等同于 Original，Normalized 模式等同于 Base Scale
            nn.init.zeros_(self.scale_mlp[-1].weight)
            nn.init.zeros_(self.scale_mlp[-1].bias)

            # 3. [关键] Normalized 模式的可学习基准 (Learnable Base)
            # 不再硬编码 50.0，初始化为 80.0 (经验舒适区)，允许梯度调整
            if self.decoupling_mode == "normalized":
                self.base_scale = nn.Parameter(torch.tensor([80.0]))

            print(
                f"Mode: {self.decoupling_mode} | Zero-Init: ON | Base Scale: {'Learnable' if hasattr(self, 'base_scale') else 'N/A'}")

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):

        logits = []

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None

        # [Probe] 初始化监控容器
        batch_probe_data = {
            "alpha": [],
            "norm": [],
            "cosine": []
        }
        # [Reg] 用于收集所有生成的 alpha 以计算 L1 Loss
        all_alpha_tensors = []
        
        last_thoughts_cache = {}

        for pass_idx in range(max_n_latents):

            if kv_cache == None:
                # first forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                # extract kv cache to reuse
                past_key_values_legacy = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if DynamicCache is not None:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values_legacy)
                else:
                    past_key_values = past_key_values_legacy

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )

                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[-1]
            
            if hasattr(outputs.past_key_values, "to_legacy_cache"):
                kv_cache = outputs.past_key_values.to_legacy_cache()
            else:
                kv_cache = outputs.past_key_values

            # [Logic] 计算 Intensity (Alpha)
            intensity_scales = None
            if self.decoupling_mode in ["residual", "normalized"]:
                # [关键] 针对 Normalized 模式，必须切断“模长反馈回路”
                # 如果是 Normalized 模式，输入给 MLP 的必须是归一化后的向量 (Direction)
                # 否则 input norm 越大 -> MLP output 越大 -> next input norm 更大 (爆炸/死锁)
                mlp_input = hidden_states
                if self.decoupling_mode == "normalized":
                    mlp_input = F.normalize(hidden_states, p=2, dim=-1).detach()  # Detach 甚至可以更彻底切断梯度，这里先只做 Normalize

                intensity_scales = self.scale_mlp(mlp_input)
                all_alpha_tensors.append(intensity_scales)

            # feedback logic
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                # 1. 原始 Hidden State
                raw_h = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

                # 2. 应用解耦与门控机制
                if self.decoupling_mode == "residual":
                    # [修复 Residual]
                    # 1. Zero-Init 保证了初始 alpha=0
                    # 2. 增加阻尼系数 (0.1)，防止复利效应导致的指数爆炸
                    alpha_raw = intensity_scales[batch_idx, token_idx - 1 - hidden_states_offset, :]

                    # 限制范围 [-0.1, 0.1]，既保留了非线性，又很安全
                    alpha = 0.1 * torch.tanh(alpha_raw)

                    final_h = raw_h * (1 + alpha)

                    batch_probe_data["alpha"].append(alpha.abs().mean().detach())

                elif self.decoupling_mode == "normalized":
                    # [修复 Normalized]
                    # 1. 之前 MLP 输入已经是 normalize 过的了，所以这里 alpha 纯粹由“语义”决定
                    # 2. 使用 base_scale * exp(alpha) 实现绝对控制
                    gate = intensity_scales[batch_idx, token_idx - 1 - hidden_states_offset, :]

                    # 计算方向
                    norm_val = torch.norm(raw_h, p=2, dim=-1, keepdim=True) + 1e-6
                    direction = raw_h / norm_val

                    # 计算最终模长
                    # 初始状态 gate=0 -> scale = base_scale (80.0)
                    # gate > 0 -> 放大; gate < 0 -> 缩小
                    # exp 保证了 scale 永远为正
                    scale = self.base_scale * torch.exp(gate)

                    final_h = direction * scale

                    batch_probe_data["alpha"].append(gate.abs().mean().detach())

                else:
                    # Original
                    final_h = raw_h
                    batch_probe_data["alpha"].append(torch.tensor(0.0, device=raw_h.device))

                # [Probe] 记录 Norm
                current_norm = torch.norm(final_h, p=2).detach()
                batch_probe_data["norm"].append(current_norm)

                # [Probe] 记录 Cosine
                if batch_idx in last_thoughts_cache:
                    prev_h = last_thoughts_cache[batch_idx]
                    cos_sim = F.cosine_similarity(final_h, prev_h, dim=0).detach()
                    batch_probe_data["cosine"].append(cos_sim)
                
                last_thoughts_cache[batch_idx] = final_h.detach()

                tensor_list[batch_idx][token_idx] = final_h

            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # final pass
        if kv_cache:
             past_key_values_legacy = [
                (
                    k[:, :, : next_compute_range[0], :],
                    v[:, :, : next_compute_range[0], :],
                )
                for k, v in kv_cache
            ]
             if DynamicCache is not None:
                 past_key_values = DynamicCache.from_legacy_cache(past_key_values_legacy)
             else:
                 past_key_values = past_key_values_legacy
        else:
             past_key_values = None

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=past_key_values,
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        
        # 基础 LM Loss
        lm_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        # [NEW] 计算 L1 Sparsity Loss
        sparsity_loss = torch.tensor(0.0, device=lm_loss.device)
        if len(all_alpha_tensors) > 0:
            # 将所有步骤的 alpha 拼在一起
            all_alphas = torch.cat(all_alpha_tensors, dim=1) # (batch, total_seq, 1)
            
            # 目标：让 alpha 趋近于 0 (稀疏)
            # 对于 Tanh Residual, alpha=0 意味着不做改变 (Identity)
            # 对于 Normalized, alpha=0 意味着 Silence
            sparsity_loss = all_alphas.abs().mean()

        total_loss = lm_loss + self.sparsity_weight * sparsity_loss

        # [Probe] 聚合统计数据
        def safe_mean(k):
            if len(batch_probe_data[k]) > 0:
                return torch.stack(batch_probe_data[k]).mean()
            return torch.tensor(0.0, device=self.embedding.weight.device)

        final_probes = {
            "probe/avg_alpha": safe_mean("alpha"),
            "probe/avg_norm": safe_mean("norm"),
            "probe/avg_cosine": safe_mean("cosine"),
            "probe/reg_loss": sparsity_loss.detach() # 记录一下正则项大小
        }

        return Outputs(loss=total_loss, inputs_embeds=inputs_embeds, logits=logits, probes=final_probes)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):
        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()
        labels = input_ids.clone()
        
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds
        else:
            return torch.tensor(tokens).view(1, -1)