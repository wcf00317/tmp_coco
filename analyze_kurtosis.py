import torch
import torch.nn as nn
import numpy as np
import json
import re
import argparse
import gc
from scipy.stats import kurtosis
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from collections import namedtuple
import matplotlib.pyplot as plt
import seaborn as sns

# 尝试导入，如果失败则手动定义以防环境问题
try:
    from coconut import Coconut
except ImportError:
    pass

# [FIX] 显式定义 namedtuple 避免 NameError
Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])

# --- 1. 鲁棒的提取与判断工具 ---
def extract_answer(text):
    text = text.replace(",", "")
    # 优先匹配 ### 后的答案
    matches = re.findall(r"###\s*(-?\d+\.?\d*)", text)
    if matches: return matches[-1]
    # 兜底匹配最后一个数字
    matches = re.findall(r'-?\d+\.?\d*', text)
    if matches: return matches[-1]
    return ""

def check_correctness(pred, gt):
    try:
        return abs(float(pred) - float(gt)) < 1e-4
    except:
        return pred.strip() == gt.strip()

# --- 2. Coconut Probe (用于捕获 Latents) ---
class CoconutProbe(nn.Module):
    def __init__(self, original_coconut_model):
        super().__init__()
        # 偷梁换柱：直接复用传入的 coconut 实例的属性
        self.base_causallm = original_coconut_model.base_causallm
        self.latent_token_id = original_coconut_model.latent_token_id
        self.embedding = original_coconut_model.embedding
        self.captured_latents = []
        # 复制 forward 逻辑... 为了代码简洁，这里假设你已经有 coconut.py
        # 实际运行时，我们可以动态给 original_coconut_model 打补丁，不用重写类

    # 这里我们使用 "Monkey Patch" 技巧动态修改 forward，比继承更灵活
    pass

def hook_forward_for_capture(model_instance):
    """
    给 Coconut 模型实例动态打补丁，插入 Latent 捕获逻辑
    """
    original_forward = model_instance.forward
    model_instance.captured_latents = [] # 绑定一个新属性

    def hijacked_forward(input_ids, attention_mask, labels=None, position_ids=None, **kwargs):
        # 清空
        # model_instance.captured_latents = [] # 注意：generate 会多次调用 forward，不能在这里清空
        
        # 也就是 coconut.py 里的逻辑，我们需要完全复制一遍来插入 hook
        # 为了避免代码太长，这里演示核心修改逻辑：
        # 我们假设用户使用的是标准 coconut.py，我们在 forward 内部是无法直接 hook 的
        # 所以必须整个替换。
        
        # --- 复制 coconut.py 的 forward 逻辑并修改 ---
        logits = []
        latent_indices = (input_ids == model_instance.latent_token_id).nonzero()
        latent_lists = [[idx[1].item() for idx in latent_indices if idx[0] == i] for i in range(input_ids.shape[0])]
        max_n_latents = max([len(l) for l in latent_lists])
        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = model_instance.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None

        for pass_idx in range(max_n_latents):
            if kv_cache == None:
                outputs = model_instance.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0] : next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0
            else:
                # [FIX for Transformers >= 4.36]
                past_key_values = tuple([(k[:, :, : next_compute_range[0], :], v[:, :, : next_compute_range[0], :]) for k, v in kv_cache])
                outputs = model_instance.base_causallm(
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

            # --- HOOK START ---
            filling_indices = [(instance_idx, mask_list[pass_idx]) for instance_idx, mask_list in enumerate(latent_lists) if len(mask_list) > pass_idx]
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                vector = hidden_states[batch_idx, token_idx - 1 - hidden_states_offset, :]
                model_instance.captured_latents.append(vector.detach().cpu())
            # --- HOOK END ---

            tensor_list = [[inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])] for batch_idx in range(inputs_embeds.shape[0])]
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                tensor_list[batch_idx][token_idx] = hidden_states[batch_idx, token_idx - 1 - hidden_states_offset, :]

            inputs_embeds = torch.stack([torch.stack(tensor_list[batch_idx]) for batch_idx in range(inputs_embeds.shape[0])])

        # Final pass
        final_past_key_values = tuple([(k[:, :, : next_compute_range[0], :], v[:, :, : next_compute_range[0], :]) for k, v in kv_cache]) if kv_cache else None
        
        outputs = model_instance.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=final_past_key_values,
            output_hidden_states=True,
        )
        logits.append(outputs.logits)
        
        model_instance.gen_forward_cnt += max_n_latents + 1
        logits = torch.cat(logits, dim=-2)
        
        loss = None
        if labels is not None: loss = torch.tensor(0.0).to(logits.device)
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    return hijacked_forward


# --- 3. 评测逻辑 ---
def eval_coconut_model(model_path, base_model_id, dataset, c_thought, tokenizer):
    print(f"\n[Phase 1] Loading Coconut Model from {model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    base_model.resize_token_embeddings(len(tokenizer))
    
    # 初始化 Coconut Wrapper
    from coconut import Coconut # 重新导入确保类定义存在
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    
    model = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    
    # 注入 Hook
    model.forward = hook_forward_for_capture(model)
    
    if torch.cuda.is_available(): model.to("cuda")
    model.eval()
    
    results = []
    
    print("Evaluating Coconut...")
    for item in tqdm(dataset):
        q_tokens = tokenizer.encode(item["question"], return_tensors="pt").to("cuda")
        k = c_thought * 2 # 假设思考步数
        latent_seq = torch.tensor([start_id] + [latent_id]*k + [end_id]).to("cuda").unsqueeze(0)
        input_ids = torch.cat([q_tokens, latent_seq], dim=1)
        
        model.captured_latents = [] # Reset hook buffer
        
        with torch.no_grad():
            output_tokens = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=64)
            
        gen_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)[len(item["question"]):].strip()
        is_correct = check_correctness(extract_answer(gen_text), item["answer"])
        
        kurt = np.nan
        if model.captured_latents:
            # Flatten all latent vectors from this inference
            vecs = torch.stack(model.captured_latents).flatten().float().cpu().numpy()
            kurt = kurtosis(vecs)
            
        results.append({"type": "Coconut", "correct": is_correct, "kurtosis": kurt, "output": gen_text})
    
    del model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    return results

def eval_cot_model(model_path, base_model_id, dataset, tokenizer):
    print(f"\n[Phase 2] Loading Standard CoT Model from {model_path}...")
    
    # 1. 先加载原始结构，不要立刻 Resize
    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    
    # 2. 加载 Checkpoint 权重
    state_dict = torch.load(model_path, map_location="cpu")
    
    # 3. 清理 Key 前缀 (处理可能的 base_causallm 前缀)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("base_causallm."):
            new_state_dict[k.replace("base_causallm.", "")] = v
        else:
            new_state_dict[k] = v
            
    # 4. [FIX] 智能 Resize：根据 Checkpoint 的权重形状来调整模型
    # 尝试寻找 embedding 权重
    vocab_size_in_checkpoint = None
    
    # GPT-2 的 embedding 键名通常是 "transformer.wte.weight"
    # Llama 的可能是 "model.embed_tokens.weight"
    possible_embed_keys = ["transformer.wte.weight", "model.embed_tokens.weight", "wte.weight"]
    
    for key in possible_embed_keys:
        if key in new_state_dict:
            vocab_size_in_checkpoint = new_state_dict[key].shape[0]
            print(f"Detected vocab size in checkpoint: {vocab_size_in_checkpoint}")
            break
            
    if vocab_size_in_checkpoint:
        # 如果 checkpoint 的大小和当前 base model 不一致，才去 resize
        if model.get_input_embeddings().weight.shape[0] != vocab_size_in_checkpoint:
            print(f"Resizing model embeddings to match checkpoint: {vocab_size_in_checkpoint}")
            model.resize_token_embeddings(vocab_size_in_checkpoint)
    else:
        # 如果找不到 embedding 权重（极少见），则回退到 tokenizer 长度，或者保持原样
        print("Warning: Could not detect vocab size from checkpoint. Assuming standard base model size.")
        # 这里我们就不 resize 了，假设它是标准模型
        
    # 5. 加载权重
    model.load_state_dict(new_state_dict, strict=False)
    
    if torch.cuda.is_available(): model.to("cuda")
    model.eval()
    
    results = []
    
    print("Evaluating Standard CoT...")
    for item in tqdm(dataset):
        # CoT Prompt: Question + \n
        prompt = item["question"] + "\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            # 注意：如果 CoT 模型词表是 50257，而 Tokenizer 是 50260，
            # 只要 prompt 里不包含那 3 个特殊 token，就不会报错。
            # 我们的 prompt 只是纯文本问题，所以是安全的。
            outputs = model.generate(
                **inputs, 
                max_new_tokens=64, 
                output_hidden_states=True, 
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        gen_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)[len(prompt):].strip()
        is_correct = check_correctness(extract_answer(gen_text), item["answer"])
        
        # Calculate Kurtosis for CoT Tokens
        cot_acts = []
        if outputs.hidden_states:
            for step_states in outputs.hidden_states:
                last_layer = step_states[-1] 
                cot_acts.append(last_layer.squeeze().flatten().cpu())
        
        kurt = np.nan
        if cot_acts:
            vecs = torch.cat(cot_acts).float().numpy()
            kurt = kurtosis(vecs)
            
        results.append({"type": "CoT", "correct": is_correct, "kurtosis": kurt, "output": gen_text})

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coconut_path", type=str, default="/mnt/bn/rai/zout/coconut/checkpoints/gsm-coconut/gsm-original-probe/checkpoint_14", help="Path to Coconut trained checkpoint")
    parser.add_argument("--cot_path", type=str, default="/mnt/bn/rai/zout/coconut/YOUR_PATH_TO_SAVE_THE_MODEL/gsm-cot/checkpoint_16", help="Path to Standard CoT trained checkpoint")
    parser.add_argument("--data_path", type=str, default="/mnt/bn/rai/zout/coconut/data/gsm_test.json")
    parser.add_argument("--base_model_id", type=str, default="openai-community/gpt2")
    parser.add_argument("--c_thought", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    # Load Data
    try:
        with open(args.data_path, 'r') as f: dataset = json.load(f)
    except:
        dataset = []
        with open(args.data_path, 'r') as f:
            for line in f: dataset.append(json.loads(line))
    dataset = dataset[:args.max_samples]

    # Setup Tokenizer (Shared)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])

    # 1. Eval Coconut
    res_coc = eval_coconut_model(args.coconut_path, args.base_model_id, dataset, args.c_thought, tokenizer)
    
    # 2. Eval CoT
    res_cot = eval_cot_model(args.cot_path, args.base_model_id, dataset, tokenizer)

    # --- 统计与展示 ---
    print("\n" + "="*50)
    print("DUAL MODEL COMPARISON REPORT")
    print("="*50)
    
    def get_stats(res):
        acc = sum(r['correct'] for r in res) / len(res)
        # Filter NaNs
        k_vals = [r['kurtosis'] for r in res if not np.isnan(r['kurtosis'])]
        avg_k = np.mean(k_vals) if k_vals else 0
        return acc, avg_k, k_vals

    acc_coc, k_coc, kv_coc = get_stats(res_coc)
    acc_cot, k_cot, kv_cot = get_stats(res_cot)
    
    print(f"Coconut Model | Acc: {acc_coc:.2%} | Avg Kurtosis: {k_coc:.2f}")
    print(f"Std CoT Model | Acc: {acc_cot:.2%} | Avg Kurtosis: {k_cot:.2f}")
    
    # 画图
    plt.figure(figsize=(10, 6))
    sns.kdeplot(kv_coc, fill=True, label=f'Coconut (Acc={acc_coc:.2%})', color='orange', alpha=0.4)
    sns.kdeplot(kv_cot, fill=True, label=f'Standard CoT (Acc={acc_cot:.2%})', color='blue', alpha=0.4)
    plt.title("Activation Kurtosis Distribution: Mature CoT vs Coconut")
    plt.xlabel("Kurtosis Value per Sample")
    plt.legend()
    plt.savefig("dual_model_comparison.png")
    print("\nSaved comparison plot to 'dual_model_comparison.png'")
    
    # Failure Analysis Check
    print("\n[Comparison of Correct Samples Only]")
    k_coc_corr = np.mean([r['kurtosis'] for r in res_coc if r['correct']])
    k_cot_corr = np.mean([r['kurtosis'] for r in res_cot if r['correct']])
    print(f"Coconut Correct Kurtosis: {k_coc_corr:.2f}")
    print(f"CoT Correct Kurtosis:     {k_cot_corr:.2f}")

if __name__ == "__main__":
    main()