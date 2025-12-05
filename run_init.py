# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from datetime import timedelta
import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig # [新增] 用于安全保存
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from probe_coconut import Coconut
from dataset import (
    get_dataset,
    get_question_latent_dataset,
    get_cot_latent_dataset,
    MyCollator,
)

import logging
from tqdm import tqdm
from copy import copy
import itertools
import os, sys, shutil
import yaml
import json
import gc
import argparse
import functools
from utils import Config, set_seed
import torch.multiprocessing as mp
try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
except ImportError:
    Qwen2DecoderLayer = None

def setup_logger(save_dir, rank):
    logger = logging.getLogger(f"coconut_rank_{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if rank == 0:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        log_file = os.path.join(save_dir, "training_log.txt")
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def worker(rank, world_size, args):
    # 环境变量设置
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(rank)

    # 初始化进程组
    init_file = "/tmp/coconut_dist_lock_fixed"
    if os.name == 'nt':
        init_method = f"file:///{init_file}"
    else:
        init_method = f"file://{init_file}"

    print(f"[Rank {rank}] Initializing process group via {init_method}...")

    try:
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=60)
        )
    except Exception as e:
        print(f"[Rank {rank}] NCCL init failed ({e}), trying GLOO...")
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=60)
        )

    # 加载配置
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    logger = setup_logger(save_dir, rank)
    if rank == 0:
        logger.info(f"Config Loaded: {config_dict}")
        logger.info(f"Save Directory: {save_dir}")
    
    torch.distributed.barrier()
    
    # 断点续训逻辑
    cur_ckpts = os.listdir(save_dir) if os.path.exists(save_dir) else []
    if len(cur_ckpts) > 0 and not configs.only_eval:
        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))

        latest_checkpoint = checkpoints[-1] if checkpoints else None
        if latest_checkpoint:
            configs.resume = int(latest_checkpoint.split("_")[1])
            load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)
            configs.load_model_path = load_dir
            if rank == 0:
                print(f"Loading from previous run epoch_{configs.resume}!")

    # 模型与 Tokenizer 加载
    model = AutoModelForCausalLM.from_pretrained(configs.model_id, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    loaded = False
    if configs.load_model_path != "None":
        saved_weights = torch.load(
            configs.load_model_path, map_location=torch.device(rank)
        )
        if configs.coconut and not any([k.startswith("base_causallm") for k in saved_weights.keys()]):
            loaded = True
            print(f"[Rank {rank}] Loaded weights directly into base model.")
            model.load_state_dict(saved_weights, strict=False)

    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        # 初始化特殊 token 的 embedding
        for token_id in [latent_id, start_id, end_id]:
            target_embedding = embeddings.weight.data[target_id]
            embeddings.weight.data[token_id] = target_embedding
            lm_head = model.lm_head
            lm_head.weight.data[token_id] = lm_head.weight.data[target_id]

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    if configs.coconut:
        d_mode = getattr(configs, "decoupling_mode", "original")
        if rank == 0:
            logger.info(f"Initializing Coconut with mode: {d_mode}")
        model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id, decoupling_mode=d_mode)

    if configs.load_model_path != "None" and not loaded:
        print(f"[Rank {rank}] Loaded weights into wrapper model.")
        model.load_state_dict(saved_weights, strict=False)

    print(f"Running FSDP on rank = {rank}, world size = {world_size}")
    model = model.to(rank)

    # FSDP Wrap Policy
    transformer_layer_cls_set = {LlamaDecoderLayer}
    if Qwen2DecoderLayer is not None:
        transformer_layer_cls_set.add(Qwen2DecoderLayer)

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls_set,
    )

    if configs.bf16:
        model.to(torch.bfloat16)

    if configs.only_eval:
        parallel_model = DDP(model, device_ids=[rank])
    else:
        parallel_model = FSDP(
            model, auto_wrap_policy=llama_auto_wrap_policy, device_id=rank
        )

    del model

    # 数据集准备
    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))]
    cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]

    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=32 if configs.debug else 100000000
    )

    if not configs.only_eval:
        base_dataset_train = get_dataset(
            configs.train_path,
            tokenizer,
            max_size=5000 if configs.debug else 100000000,
            data_ratio=getattr(configs, "train_ratio")
        )

    max_new_tokens = 64 if "gsm" in configs.val_path else 128
    
    if configs.reset_optimizer:
        optimizer = None
    else:
        optimizer = optim.AdamW(
            parallel_model.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )

    best_acc = 0
    total_train_steps = 0
    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    # ================= 训练循环 =================
    for epoch in range(configs.resume, configs.num_epochs):

        scheduled_stage = (
            0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage
        )
        
        # 验证集 Dataset (Generation)
        dataset_gen_val = get_question_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            configs,
            start_id,
            latent_id,
            end_id,
            no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
        )

        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            batch_size=1,
            collate_fn=collator,
            sampler=DistributedSampler(dataset_gen_val, shuffle=False),
        )

        # ------------------ Training Phase ------------------
        if not configs.only_eval:
            dataset_train = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_train,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
                shuffle=True,
            )

            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_train, shuffle=True),
            )

            # Loss Validation Loader
            dataset_loss_val = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
            )

            valid_loss_dataloader = torch.utils.data.DataLoader(
                dataset_loss_val,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_loss_val, shuffle=False),
            )

            if configs.reset_optimizer:
                del optimizer
                optimizer = optim.AdamW(
                    parallel_model.parameters(),
                    lr=configs.lr,
                    weight_decay=configs.weight_decay,
                )

            parallel_model.train()
            
            if rank == 0:
                total_length = len(train_dataloader) // configs.gradient_accumulation_steps
                pbar = tqdm(
                    colour="blue",
                    desc=f"Training Epoch: {epoch + 1}",
                    total=total_length,
                    dynamic_ncols=True,
                )

            for step, batch in enumerate(train_dataloader):
                total_train_steps += 1
                batch = {key: batch[key].to(rank) for key in batch.keys() if key != "idx"}

                is_probe_step = (total_train_steps % 1000 == 0)
                outputs = parallel_model(**batch, compute_probes=is_probe_step)
                
                loss = outputs.loss / configs.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    if rank == 0: pbar.update(1)

                # Logging
                if rank == 0 and (step % 100 == 0 or is_probe_step):
                    current_loss = loss.detach().float().item() * configs.gradient_accumulation_steps
                    log_data = {"epoch": epoch + 1, "step": total_train_steps, "loss": round(current_loss, 4)}
                    if hasattr(outputs, "probes") and outputs.probes:
                        for k, v in outputs.probes.items():
                            val = v.item() if isinstance(v, torch.Tensor) else v
                            if "rank" in k or "entropy" in k: 
                                if val != 0: log_data[k.replace("probe/", "")] = round(val, 4)
                            else:
                                log_data[k.replace("probe/", "")] = round(val, 4)
                    logger.info(json.dumps(log_data))
                
                if rank == 0:
                    pbar.set_description(f"Epoch {epoch + 1} | Loss: {loss.item() * configs.gradient_accumulation_steps:.4f}")

            if rank == 0: pbar.close()
            dist.barrier()

            # Save Latest Checkpoint
            if not configs.save_only_improve and not configs.debug:
                full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(parallel_model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                    states = parallel_model.state_dict()
                
                if rank == 0:
                    torch.save(states, os.path.join(save_dir, f"checkpoint_{epoch + 1}"))
                    logger.info(f"Checkpoint saved: checkpoint_{epoch + 1}")
                    del states
                dist.barrier()
                gc.collect()

            # Val Loss Calculation
            total_loss = torch.tensor(0.0, device=rank)
            with torch.no_grad():
                parallel_model.eval()
                for batch in valid_loss_dataloader:
                    batch = {k: v.to(rank) for k, v in batch.items() if k != "idx"}
                    outputs = parallel_model(**batch, compute_probes=False)
                    total_loss += outputs.loss

            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            if rank == 0:
                # 这里的 len 需要注意是全局还是局部，简化起见这里用近似值
                eval_loss = total_loss.item() / (len(valid_loss_dataloader) * world_size)
                logger.info(f"Evaluation Loss (Epoch {epoch + 1}): {eval_loss}")

        # ------------------ Generation Validation Phase ------------------
        # [关键修复]：移除 if rank == 0，所有卡必须参与 Generate 避免死锁
        dist.barrier()
        
        local_cor = torch.tensor(0.0, device=rank)
        local_cor_cot = torch.tensor(0.0, device=rank)
        local_total = torch.tensor(0.0, device=rank)

        if rank == 0:
            pbar = tqdm(colour="blue", desc="Test Accuracy", total=len(valid_gen_dataloader)*world_size)

        with torch.no_grad():
            parallel_model.eval()
            for idx, batch in enumerate(valid_gen_dataloader):
                test_idx = batch["idx"][0]
                batch = {
                    k: v.to(rank) for k, v in batch.items() 
                    if v is not None and k not in ["idx", "position_ids"]
                }

                # FSDP 要求所有 rank 同时调用 generate
                outputs = parallel_model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    synced_gpus=True, 
                )

                # 后处理与统计 (Local)
                text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 容错解析：如果找不到分隔符，使用整个文本的最后部分
                if "#" in text_output:
                    answer_output = text_output.split("#")[-1].replace(",", "").strip()
                    cot_output = ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()
                else:
                    answer_output = text_output.strip().split()[-1] if text_output.strip() else ""
                    cot_output = ""

                answer = answers_val[test_idx.cpu().item()]
                answer_cot = cot_val[test_idx.cpu().item()]

                if answer_output == answer:
                    local_cor += 1
                if cot_output == answer_cot:
                    local_cor_cot += 1
                local_total += 1

                # 仅 Rank 0 打印样例
                if idx < 1 and rank == 0:
                    logger.info(f"[Sample] GT: {answer} | Pred: {answer_output}")

                if rank == 0: pbar.update(world_size)

        if rank == 0: pbar.close()

        # [关键修复]：汇总所有卡的统计结果
        dist.all_reduce(local_cor, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_cor_cot, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total, op=dist.ReduceOp.SUM)

        global_acc = (local_cor / local_total).item() if local_total.item() > 0 else 0.0
        global_cot_acc = (local_cor_cot / local_total).item() if local_total.item() > 0 else 0.0

        if rank == 0:
            logger.info(f"Epoch {epoch+1} Accuracy: {global_acc:.4f} (Best: {best_acc:.4f})")
            logger.info(f"Epoch {epoch+1} CoT EM: {global_cot_acc:.4f}")

        if configs.only_eval:
            break

        dist.barrier()

        # [关键修复]：使用同步后的 global_acc 判断是否保存，并使用安全保存上下文
        if (global_acc > best_acc and configs.save_only_improve and not configs.debug):
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(parallel_model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                states = parallel_model.state_dict()

            if rank == 0:
                torch.save(states, os.path.join(save_dir, f"checkpoint_{epoch + 1}"))
                logger.info(f"New best model saved! Acc: {global_acc:.4f}")
                del states
            
            best_acc = global_acc
            dist.barrier()
            gc.collect()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")
    args, unknown = parser.parse_known_args()

    world_size = torch.cuda.device_count()
    print(f"Detected {world_size} GPUs. Using FileStore for init.")

    lock_path = "/tmp/coconut_dist_lock_fixed"
    if os.path.exists(lock_path):
        try:
            os.remove(lock_path)
        except:
            pass

    mp.spawn(worker, args=(world_size, args), nprocs=world_size)