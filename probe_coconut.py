# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel
import logging # [新增]

# 获取 run_init.py 初始化的 logger，确保写入同一个 txt
logger = logging.getLogger("coconut_rank_0")

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits", "probes"])
MAX_N_LATENT = 8

# ================= [工具类修复] =================
class MetricCalculator:
    @staticmethod
    def compute_entropy(attention_matrix):
        # [诊断] 必须先判空
        if attention_matrix is None:
            print("[Metric Error] Input attention_matrix is None!")
            return torch.tensor(0.0)
            
        try:
            if torch.isnan(attention_matrix).any():
                print("[Metric Warn] Attention matrix contains NaN!")
                return torch.tensor(0.0, device=attention_matrix.device)

            entropy = -torch.sum(attention_matrix * torch.log(attention_matrix + 1e-9), dim=-1)
            return entropy.mean()
        except Exception as e:
            # [诊断] 强制 Print 到控制台
            print(f"[Metric Error] Entropy failed: {e}")
            return torch.tensor(0.0)

    @staticmethod
    def compute_effective_rank(attention_matrix):
        if attention_matrix is None:
            return torch.tensor(0.0)

        try:
            # 1. 强转 float32
            matrix = attention_matrix.float()

            if torch.isnan(matrix).any() or torch.isinf(matrix).any():
                print(f"[Metric Error] Matrix has NaN/Inf!")
                return torch.tensor(0.0, device=attention_matrix.device)

            # 2. SVD 计算
            with torch.no_grad():
                s = torch.linalg.svdvals(matrix)
                s_sum = s.sum(dim=-1, keepdim=True)
                p = s / (s_sum + 1e-9)
                entropy = -torch.sum(p * torch.log(p + 1e-9), dim=-1)
                er = torch.exp(entropy)
                return er.mean()

        except Exception as e:
            # [诊断] 强制 Print 到控制台
            print(f"[Metric Error] Rank Calculation Failed: {e}")
            return torch.tensor(0.0)

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
        self.norm_scale_factor = 50.0
        
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
        max_n_latents = max([len(l) for l in latent_lists])
        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)
        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None
        
        batch_probe_data = {"alpha": [], "norm": [], "cosine": []}
        all_alpha_tensors = []
        last_thoughts_cache = {}
        advanced_metrics = {}

        for pass_idx in range(max_n_latents):
            if kv_cache == None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0] : next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    output_hidden_states=True,
                    output_attentions=compute_probes, # 传递开关
                )
                hidden_states_offset = 0
            else:
                past_key_values_legacy = [(k[:, :, : next_compute_range[0], :], v[:, :, : next_compute_range[0], :]) for k, v in kv_cache]
                if DynamicCache is not None:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values_legacy)
                else:
                    past_key_values = past_key_values_legacy

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    output_attentions=compute_probes, # 传递开关
                )
                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)
            
            # [诊断逻辑] 为什么是 0.0？
            if compute_probes:
                if outputs.attentions is None:
                    # 如果这行出现在日志里，说明模型不支持返回 attention
                    if torch.distributed.get_rank() == 0:
                        logger.warning(f"[Probe Fail] Step {pass_idx}: outputs.attentions is None!")
                else:
                    # 取最后一层 Attention
                    last_attn = outputs.attentions[-1]
                    
                    ent = MetricCalculator.compute_entropy(last_attn)
                    rank_val = MetricCalculator.compute_effective_rank(last_attn)
                    
                    if "entropy" not in advanced_metrics: advanced_metrics["entropy"] = []
                    if "rank" not in advanced_metrics: advanced_metrics["rank"] = []
                    
                    advanced_metrics["entropy"].append(ent)
                    advanced_metrics["rank"].append(rank_val)

            next_compute_range = (next_compute_range[1], (input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1))
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
                all_alpha_tensors.append(intensity_scales)

            filling_indices = [(instance_idx, mask_list[pass_idx]) for instance_idx, mask_list in enumerate(latent_lists) if len(mask_list) > pass_idx]
            tensor_list = [[inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])] for batch_idx in range(inputs_embeds.shape[0])]

            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                raw_h = hidden_states[batch_idx, token_idx - 1 - hidden_states_offset, :]
                
                if self.decoupling_mode == "residual":
                    alpha_raw = intensity_scales[batch_idx, token_idx - 1 - hidden_states_offset, :]
                    alpha = torch.tanh(alpha_raw) 
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
                if batch_idx in last_thoughts_cache:
                    prev_h = last_thoughts_cache[batch_idx]
                    cos_sim = F.cosine_similarity(final_h, prev_h, dim=0).detach()
                    batch_probe_data["cosine"].append(cos_sim)
                last_thoughts_cache[batch_idx] = final_h.detach()
                tensor_list[batch_idx][token_idx] = final_h

            inputs_embeds = torch.stack([torch.stack(tensor_list[batch_idx]) for batch_idx in range(inputs_embeds.shape[0])])

        if kv_cache:
             past_key_values_legacy = [(k[:, :, : next_compute_range[0], :], v[:, :, : next_compute_range[0], :]) for k, v in kv_cache]
             if DynamicCache is not None:
                 past_key_values = DynamicCache.from_legacy_cache(past_key_values_legacy)
             else:
                 past_key_values = past_key_values_legacy
        else:
             past_key_values = None

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
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
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        sparsity_loss = torch.tensor(0.0, device=lm_loss.device)
        if len(all_alpha_tensors) > 0 and self.sparsity_weight > 0:
            all_alphas = torch.cat(all_alpha_tensors, dim=1)
            sparsity_loss = all_alphas.abs().mean()
        total_loss = lm_loss + self.sparsity_weight * sparsity_loss

        def safe_mean(k, source_dict=batch_probe_data):
            if k in source_dict and len(source_dict[k]) > 0:
                return torch.stack(source_dict[k]).mean()
            return torch.tensor(0.0, device=self.embedding.weight.device)

        final_probes = {
            "probe/avg_alpha": safe_mean("alpha"),
            "probe/avg_norm": safe_mean("norm"),
            "probe/avg_cosine": safe_mean("cosine"),
            "probe/reg_loss": sparsity_loss.detach(),
            "probe/avg_rank": safe_mean("rank", advanced_metrics),
            "probe/avg_entropy": safe_mean("entropy", advanced_metrics)
        }

        return Outputs(loss=total_loss, inputs_embeds=inputs_embeds, logits=logits, probes=final_probes)

    def train(self): self.base_causallm.train()
    def eval(self): self.base_causallm.eval()
    def generate(self, input_ids, attention_mask, max_new_tokens=16, output_embedding=False, synced_gpus=False, **kwargs):
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
