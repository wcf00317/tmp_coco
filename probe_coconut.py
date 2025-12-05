# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel
import logging
import torch.distributed as dist

# 获取 run_init.py 初始化的 logger
logger = logging.getLogger("coconut_rank_0")

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits", "probes"])
MAX_N_LATENT = 8

class MetricCalculator:
    @staticmethod
    def compute_entropy(attention_matrix):
        if attention_matrix is None:
            return torch.tensor(0.0)
        # [优化] 移除 isnan 检查，直接计算，利用 nan_to_num 兜底
        # 避免 CPU-GPU 同步阻塞
        safe_attn = torch.nan_to_num(attention_matrix, nan=1e-9)
        entropy = -torch.sum(safe_attn * torch.log(safe_attn + 1e-9), dim=-1)
        return entropy.mean()

    @staticmethod
    def compute_effective_rank(attention_matrix):
        if attention_matrix is None:
            return torch.tensor(0.0)
        # [优化] 移除 isnan/isinf 检查
        matrix = attention_matrix.float()
        safe_matrix = torch.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        try:
            # SVD 可能会失败，但这通常是极其罕见的
            s = torch.linalg.svdvals(safe_matrix)
            s_sum = s.sum(dim=-1, keepdim=True)
            p = s / (s_sum + 1e-9)
            entropy = -torch.sum(p * torch.log(p + 1e-9), dim=-1)
            er = torch.exp(entropy)
            return er.mean()
        except:
            return torch.tensor(0.0, device=attention_matrix.device)

class Coconut(nn.Module):
    def __init__(
            self,
            base_causallm,
            latent_token_id,
            start_latent_id,
            end_latent_id,
            eos_token_id,
            decoupling_mode="original",
    ):
        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.decoupling_mode = decoupling_mode
        self.sparsity_weight = 0.0
        
        # [新增] 余弦相似度 Loss，用于惩罚偷懒 (Identity Mapping)
        # margin=0.9 表示如果相似度 > 0.9 则产生 loss
        self.cos_loss_fct = nn.CosineEmbeddingLoss(margin=0.9)

        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

        if self.decoupling_mode in ["residual", "normalized"]:
            hidden_size = self.embedding.weight.shape[1]
            self.scale_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1)
            )
            nn.init.zeros_(self.scale_mlp[-1].weight)
            nn.init.zeros_(self.scale_mlp[-1].bias)

            if self.decoupling_mode == "normalized":
                self.base_scale = nn.Parameter(torch.tensor([80.0]))

    def forward(self, input_ids, attention_mask, labels, position_ids, compute_probes=False, **kwargs):
        logits = []
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [[idx[1].item() for idx in latent_indices if idx[0] == i] for i in range(input_ids.shape[0])]
        local_max = max([len(l) for l in latent_lists]) if latent_lists else 0
        local_max_tensor = torch.tensor(local_max, device=input_ids.device)

        if dist.is_initialized():
            dist.all_reduce(local_max_tensor, op=dist.ReduceOp.MAX)
        
        max_n_latents = local_max_tensor.item()
        
        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)
        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None
        # [优化] 预分配 Tensor 列表，避免 append 操作
        batch_probe_data = {"alpha": [], "norm": [], "cosine": []}
        advanced_metrics = {"entropy": [], "rank": []}
        
        # [新增] 用于累计正则化 Loss
        reg_cos_loss = torch.tensor(0.0, device=input_ids.device)

        for pass_idx in range(max_n_latents):
            if kv_cache == None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]: next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0]: next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]: next_compute_range[1]],
                    output_hidden_states=True,
                    output_attentions=compute_probes,
                )
                hidden_states_offset = 0
            else:
                past_key_values_legacy = [(k[:, :, : next_compute_range[0], :], v[:, :, : next_compute_range[0], :]) for k, v in kv_cache]
                if DynamicCache is not None:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values_legacy)
                else:
                    past_key_values = past_key_values_legacy

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]: next_compute_range[1], :],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]: next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    output_attentions=compute_probes,
                )
                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)

            # [优化] Probe 计算逻辑：完全无阻塞
            if compute_probes and outputs.attentions is not None:
                valid_attentions = [a for a in outputs.attentions if a is not None]
                if len(valid_attentions) > 0:
                    last_attn = valid_attentions[-1]
                    # 不再使用 no_grad，因为我们只是为了看指标，不反向传播
                    # 但在这里使用 detach() 更安全
                    ent = MetricCalculator.compute_entropy(last_attn.detach())
                    rank_val = MetricCalculator.compute_effective_rank(last_attn.detach())
                    advanced_metrics["entropy"].append(ent)
                    advanced_metrics["rank"].append(rank_val)

            next_compute_range = (next_compute_range[1],
                                  (input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1))
            
            hidden_states = outputs.hidden_states[-1]

            if hasattr(outputs.past_key_values, "to_legacy_cache"):
                kv_cache = outputs.past_key_values.to_legacy_cache()
            else:
                kv_cache = outputs.past_key_values

            intensity_scales = None
            if self.decoupling_mode in ["residual", "normalized"]:
                mlp_input = hidden_states
                if self.decoupling_mode == "normalized":
                    mlp_input = F.normalize(hidden_states, p=2, dim=-1).detach()
                intensity_scales = self.scale_mlp(mlp_input)

            filling_indices = [(instance_idx, mask_list[pass_idx]) for instance_idx, mask_list in
                               enumerate(latent_lists) if len(mask_list) > pass_idx]
            
            # 使用列表推导式构建 tensor_list，避免 inplace 操作
            tensor_list = [[inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])] for batch_idx in range(inputs_embeds.shape[0])]

            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                raw_h = hidden_states[batch_idx, token_idx - 1 - hidden_states_offset, :]
                
                # [关键优化] 获取上一轮的输入向量，用于计算 Cosine Regularization
                prev_h = tensor_list[batch_idx][token_idx]

                # 1. 计算 Cosine 正则 (强制不相似)
                # 使用 nan_to_num 确保无 NaN 传入导致崩溃，且完全在 GPU 上执行
                safe_raw_h = torch.nan_to_num(raw_h, nan=0.0)
                safe_prev_h = torch.nan_to_num(prev_h.detach(), nan=0.0) # detach prev_h
                
                curr_cos_loss = self.cos_loss_fct(
                    safe_raw_h.unsqueeze(0), 
                    safe_prev_h.unsqueeze(0), 
                    target=torch.tensor([-1], device=input_ids.device)
                )
                reg_cos_loss += curr_cos_loss

                if self.decoupling_mode == "residual":
                    alpha_raw = intensity_scales[batch_idx, token_idx - 1 - hidden_states_offset, :]
                    # 限制 alpha 范围，防止梯度爆炸
                    alpha = 0.05 * torch.tanh(alpha_raw)
                    final_h = raw_h * (1 + alpha)
                    batch_probe_data["alpha"].append(alpha.abs().mean().detach())
                elif self.decoupling_mode == "normalized":
                    gate = intensity_scales[batch_idx, token_idx - 1 - hidden_states_offset, :]
                    norm_val = torch.norm(raw_h, p=2, dim=-1, keepdim=True) + 1e-6
                    direction = raw_h / norm_val
                    scale = self.base_scale * torch.exp(gate)
                    final_h = direction * scale
                    batch_probe_data["alpha"].append(gate.abs().mean().detach())
                else:
                    final_h = raw_h
                    batch_probe_data["alpha"].append(torch.tensor(0.0, device=raw_h.device))

                current_norm = torch.norm(final_h, p=2).detach()
                batch_probe_data["norm"].append(current_norm)
                
                # 计算 Cosine Similarity 仅用于记录日志，使用 detach 避免图计算
                cos_sim = F.cosine_similarity(final_h, prev_h.detach(), dim=0)
                batch_probe_data["cosine"].append(cos_sim)
                
                tensor_list[batch_idx][token_idx] = final_h

            inputs_embeds = torch.stack(
                [torch.stack(tensor_list[batch_idx]) for batch_idx in range(inputs_embeds.shape[0])])

        # Final Pass
        if kv_cache:
            past_key_values_legacy = [(k[:, :, : next_compute_range[0], :], v[:, :, : next_compute_range[0], :]) for k, v in kv_cache]
            if DynamicCache is not None:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values_legacy)
            else:
                past_key_values = past_key_values_legacy
        else:
            past_key_values = None

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]: next_compute_range[1], :],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0]: next_compute_range[1]],
            past_key_values=past_key_values,
            output_hidden_states=True,
        )
        logits.append(outputs.logits)
        self.gen_forward_cnt += max_n_latents + 1
        logits = torch.cat(logits, dim=-2)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # [关键修改] 将正则化 Loss 加入总 Loss
        # 权重设为 0.1，除以 max_n_latents 进行归一化
        total_loss = lm_loss
        if max_n_latents > 0:
            total_loss += 0.1 * (reg_cos_loss / max_n_latents)

        def safe_mean(k, source_dict):
            if k in source_dict and len(source_dict[k]) > 0:
                return torch.stack(source_dict[k]).mean()
            return torch.tensor(0.0, device=self.embedding.weight.device)

        final_probes = {
            "probe/avg_alpha": safe_mean("alpha", batch_probe_data),
            "probe/avg_norm": safe_mean("norm", batch_probe_data),
            "probe/avg_cosine": safe_mean("cosine", batch_probe_data),
            "probe/reg_loss": reg_cos_loss.detach(),
            "probe/avg_rank": safe_mean("rank", advanced_metrics),
            "probe/avg_entropy": safe_mean("entropy", advanced_metrics)
        }

        return Outputs(loss=total_loss, inputs_embeds=inputs_embeds, logits=logits, probes=final_probes)

    def train(self, mode=True): 
        super().train(mode)
        self.base_causallm.train(mode)

    def eval(self): 
        super().eval()
        self.base_causallm.eval()

    def generate(self, input_ids, attention_mask, max_new_tokens=16, output_embedding=False, synced_gpus=False, **kwargs):
        # Generate 保持不变，注意这里的 forward 调用也会走上面的逻辑，但通常 generate 不算 loss，所以安全
        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        tokens = input_ids[0].detach().tolist()
        labels = input_ids.clone()
        outputs = self.forward(input_ids, torch.ones_like(input_ids), labels, torch.arange(0, input_ids.shape[1], device=input_ids.device).reshape(1, -1))
        inputs_embeds = outputs.inputs_embeds
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(torch.tensor(next_token, device=input_ids.device)).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id: break
            tokens.append(next_token)
            new_token_embed = self.embedding(torch.tensor(next_token, device=input_ids.device)).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)
        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)
        if output_embedding: return torch.tensor(tokens).view(1, -1), new_inputs_embeds
        else: return torch.tensor(tokens).view(1, -1)