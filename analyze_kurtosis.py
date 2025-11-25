import torch
import torch.nn as nn
import numpy as np
import json
import re
import argparse
from scipy.stats import kurtosis
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from coconut import Coconut, Outputs

# --- 1. 更加鲁棒的答案提取器 ---
def extract_answer(text):
    """提取最终答案"""
    text = text.replace(",", "")
    matches = re.findall(r"###\s*(-?\d+\.?\d*)", text) # 优先找 ### 后的答案
    if matches:
        return matches[-1]
    matches = re.findall(r'-?\d+\.?\d*', text) # 兜底找最后一个数字
    if matches:
        return matches[-1]
    return ""

def check_correctness(pred, gt):
    try:
        return abs(float(pred) - float(gt)) < 1e-4
    except:
        return pred.strip() == gt.strip()

class CoconutProbe(Coconut):
    def __init__(self, base_causallm, latent_token_id, start_latent_id, end_latent_id, eos_token_id):
        super().__init__(base_causallm, latent_token_id, start_latent_id, end_latent_id, eos_token_id)
        self.captured_latents = [] 

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        self.captured_latents = []
        
        logits = []
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [[idx[1].item() for idx in latent_indices if idx[0] == i] for i in range(input_ids.shape[0])]
        max_n_latents = max([len(l) for l in latent_lists])
        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None

        for pass_idx in range(max_n_latents):
            if kv_cache == None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0] : next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0
            else:
                past_key_values = tuple([
                    (k[:, :, : next_compute_range[0], :], v[:, :, : next_compute_range[0], :])
                    for k, v in kv_cache
                ])
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)
            next_compute_range = (next_compute_range[1], (input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1))
            
            hidden_states = outputs.hidden_states[-1] 
            kv_cache = outputs.past_key_values

            # Hook
            filling_indices = [(instance_idx, mask_list[pass_idx]) for instance_idx, mask_list in enumerate(latent_lists) if len(mask_list) > pass_idx]
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                vector = hidden_states[batch_idx, token_idx - 1 - hidden_states_offset, :]
                self.captured_latents.append(vector.detach().cpu())

            tensor_list = [[inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])] for batch_idx in range(inputs_embeds.shape[0])]
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                tensor_list[batch_idx][token_idx] = hidden_states[batch_idx, token_idx - 1 - hidden_states_offset, :]

            inputs_embeds = torch.stack([torch.stack(tensor_list[batch_idx]) for batch_idx in range(inputs_embeds.shape[0])])

        # Final pass
        final_past_key_values = tuple([
            (k[:, :, : next_compute_range[0], :], v[:, :, : next_compute_range[0], :])
            for k, v in kv_cache
        ]) if kv_cache else None

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=final_past_key_values,
            output_hidden_states=True,
        )
        logits.append(outputs.logits)
        
        self.gen_forward_cnt += max_n_latents + 1
        logits = torch.cat(logits, dim=-2)
        
        loss = None
        if labels is not None:
             loss = torch.tensor(0.0).to(logits.device)

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/mnt/bn/rai/zout/coconut/checkpoints/gsm-coconut/gsm-coconut/checkpoint_6")
    parser.add_argument("--data_path", type=str, default="/mnt/bn/rai/zout/coconut/data/gsm_test.json")
    parser.add_argument("--base_model_id", type=str, default="openai-community/gpt2")
    parser.add_argument("--c_thought", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    # Load Data
    try:
        with open(args.data_path, 'r') as f:
            dataset = json.load(f)
    except:
        dataset = []
        with open(args.data_path, 'r') as f:
            for line in f:
                dataset.append(json.loads(line))
    dataset = dataset[:args.max_samples]
    
    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_id)
    base_model.resize_token_embeddings(len(tokenizer))
    
    state_dict = torch.load(args.model_path, map_location="cpu")
    coconut_model = CoconutProbe(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    coconut_model.load_state_dict(state_dict, strict=False)
    coconut_model.eval()
    if torch.cuda.is_available(): coconut_model.to("cuda")

    stats = {
        "coconut": {"correct": 0, "total": 0, "failures": []},
        "cot": {"correct": 0, "total": 0}
    }

    print(f"Comparing Coconut vs Standard CoT (Fair Mode) on {len(dataset)} samples...")
    
    for item in tqdm(dataset):
        question = item["question"]
        gt = item["answer"]
        
        # --- 1. Coconut Evaluation ---
        k = args.c_thought * 2
        # Input: Question <start> [LATENT] <end>
        q_tokens = tokenizer.encode(question, return_tensors="pt").to(coconut_model.base_causallm.device)
        latent_seq = torch.tensor([start_id] + [latent_id]*k + [end_id]).to(q_tokens.device).unsqueeze(0)
        input_ids = torch.cat([q_tokens, latent_seq], dim=1)
        
        with torch.no_grad():
            output_tokens = coconut_model.generate(
                input_ids=input_ids, 
                attention_mask=torch.ones_like(input_ids), 
                max_new_tokens=64
            )
        
        full_text_coc = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        # 假设生成内容在 question 之后
        gen_coc = full_text_coc[len(question):].strip()
        pred_coc = extract_answer(gen_coc)
        is_correct_coc = check_correctness(pred_coc, gt)
        
        if coconut_model.captured_latents:
            k_val = kurtosis(torch.stack(coconut_model.captured_latents).flatten().float().cpu().numpy())
            if not is_correct_coc:
                stats["coconut"]["failures"].append({
                    "kurtosis": k_val,
                    "question": question,
                    "output": gen_coc,
                    "gt": gt
                })
        
        stats["coconut"]["total"] += 1
        if is_correct_coc: stats["coconut"]["correct"] += 1

        # --- 2. Standard CoT Evaluation (Corrected) ---
        # Input: Question + "\n" (Training format match)
        # 不要加 "Let's think step by step"
        cot_input_text = question + "\n"
        cot_inputs = tokenizer(cot_input_text, return_tensors="pt").to(coconut_model.base_causallm.device)
        
        with torch.no_grad():
            # 使用 base_causallm 直接生成，避开 latent 逻辑
            cot_outputs = coconut_model.base_causallm.generate(
                **cot_inputs,
                max_new_tokens=64,
                pad_token_id=tokenizer.eos_token_id
            )
            
        full_text_cot = tokenizer.decode(cot_outputs[0], skip_special_tokens=True)
        gen_cot = full_text_cot[len(question):].strip()
        pred_cot = extract_answer(gen_cot)
        is_correct_cot = check_correctness(pred_cot, gt)
        
        stats["cot"]["total"] += 1
        if is_correct_cot: stats["cot"]["correct"] += 1

    # --- 报告 ---
    print("\n" + "="*50)
    print("FAIR COMPARISON RESULTS")
    print("="*50)
    print(f"Coconut Accuracy:      {stats['coconut']['correct']/stats['coconut']['total']:.2%}")
    print(f"Standard CoT Accuracy: {stats['cot']['correct']/stats['cot']['total']:.2%}")
    
    print("\n" + "="*50)
    print("SMOOTH FAILURES DEEP DIVE (Lowest Kurtosis)")
    print("="*50)
    
    failures = stats["coconut"]["failures"]
    failures.sort(key=lambda x: x["kurtosis"])
    
    for i, f in enumerate(failures[:3]):
        print(f"\n[Case {i+1}] Kurtosis: {f['kurtosis']:.2f}")
        print(f"Q: {f['question'][:60]}...")
        print(f"GT: {f['gt']}")
        print(f"Output: {f['output'].replace(chr(10), ' ')}")
        # 简单人工检查逻辑
        print("Analysis: Check the equations in the output carefully.")

if __name__ == "__main__":
    main()