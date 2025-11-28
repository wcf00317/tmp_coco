# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

# [修改] 移除 wandb
# import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
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
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(rank)

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
            rank=rank
        )
    except Exception as e:
        print(f"[Rank {rank}] NCCL init failed ({e}), trying GLOO...")
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            world_size=world_size,
            rank=rank
        )

    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if rank == 0:
        print("Config:", config_dict)

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
    cur_ckpts = os.listdir(save_dir)

    if len(cur_ckpts) > 0 and not configs.only_eval:
        if rank == 0:
            print(
                f"Warning: found previous run and gonna resume from that. the inputted `resume` argument is ignored!"
            )

        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))

        latest_checkpoint = checkpoints[-1] if checkpoints else None
        if latest_checkpoint:
            configs.resume = int(latest_checkpoint.split("_")[1])
            load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)

            configs.load_model_path = load_dir
            print(f"Loading from previous run epoch_{configs.resume}!")

    elif configs.resume != 0:
        if configs.load_model_path == "None":
            print(
                f"Warning: you want to skip the first {configs.resume} but you are not loading any existing checkpoint!"
            )
        print(
            f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs"
        )

    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
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

        if configs.coconut and not any(
                [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

        elif not configs.coconut and any(
                [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            raise ValueError("Cannot load coconut model weights into a causallm model")

        elif configs.coconut and any(
                [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            pass

        else:
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
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
        model.load_state_dict(saved_weights, strict=False)

    if rank == 0:
        logger.info(f"Running FSDP on rank = {rank}")

    if configs.load_model_path != "None" and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

    print(f"Running FSDP on rank = {rank}, world size = {world_size}")
    model = model.to(rank)

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer
        },
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

    if rank == 0:
        print(parallel_model)

    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [
        d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))
    ]
    cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]

    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=32 if configs.debug else 100000000
    )

    if not configs.only_eval:
        base_dataset_train = get_dataset(
            configs.train_path, tokenizer, max_size=5000 if configs.debug else 100000000
        )

    if "gsm" in configs.val_path:
        max_new_tokens = 64
    else:
        max_new_tokens = 128

    total_train_steps = 0

    # [修改] 移除了 wandb 初始化逻辑

    if configs.reset_optimizer:
        optimizer = None
    else:
        optimizer = optim.AdamW(
            parallel_model.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )

    best_acc = 0

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    for epoch in range(configs.resume, configs.num_epochs):

        scheduled_stage = (
            0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage
        )
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

            parallel_model.module.train()

            total_length = len(train_dataloader) // configs.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch + 1}",
                total=total_length,
                dynamic_ncols=True,
            )

            for step, batch in enumerate(train_dataloader):

                # [修改] 移除了 text_table 填充代码

                total_train_steps += 1
                batch = {
                    key: batch[key].to(rank) for key in batch.keys() if key != "idx"
                }

                # [新增] 计算开关：每 1000 步计算一次 Rank/Entropy
                is_probe_step = (total_train_steps % 1000 == 0)

                # [修改] 传入参数
                outputs = parallel_model(**batch, compute_probes=is_probe_step)

                loss = outputs.loss / configs.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                        train_dataloader
                ) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                # [修改] 如果是 probe step 也强制打印，否则看不到结果
                if rank == 0 and (step % 100 == 0 or step == len(train_dataloader) - 1 or is_probe_step):
                    current_loss = loss.detach().float().item() * configs.gradient_accumulation_steps

                    log_data = {
                        "epoch": epoch + 1,
                        "step": total_train_steps,
                        "loss": round(current_loss, 4)
                    }

                    if hasattr(outputs, "probes") and outputs.probes is not None:
                        for k, v in outputs.probes.items():
                            val = v.item() if isinstance(v, torch.Tensor) else v
                            key_name = k.replace("probe/", "")
                            # 跳过 0 值 (Rank/Entropy 在非采样步是 0)
                            if ("rank" in key_name or "entropy" in key_name) and val == 0: continue

                            log_data[key_name] = round(val, 4) if isinstance(val, float) else val

                    logger.info(json.dumps(log_data))

                pbar.set_description(
                    f"Training Epoch: {epoch + 1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}"
                )
            pbar.close()
            dist.barrier()

            if (
                    not configs.save_only_improve
                    and not configs.debug
                    and not configs.only_eval
            ):
                states = parallel_model.state_dict()
                if rank == 0:
                    torch.save(
                        states, os.path.join(save_dir, f"checkpoint_{epoch + 1}")
                    )
                    logger.info(f"Checkpoint saved: checkpoint_{epoch + 1}")

                dist.barrier()
                del states
                gc.collect()
                torch.cuda.empty_cache()

            # val loss
            total_loss = 0

            # [新增] 验证集指标累积
            val_probes_accum = {}
            val_steps = 0

            with torch.no_grad():
                parallel_model.module.eval()
                for step, batch in enumerate(valid_loss_dataloader):

                    batch = {
                        key: batch[key].to(rank) for key in batch.keys() if key != "idx"
                    }

                    # [修改] 验证集全程开启 Probes
                    outputs = parallel_model(**batch, compute_probes=True)
                    loss = outputs.loss
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    total_loss += loss.item() / world_size

                    # [新增] 缩进正确的 Probe 统计
                    if hasattr(outputs, "probes") and outputs.probes is not None:
                        if rank == 0:
                            for k, v in outputs.probes.items():
                                val = v.item() if isinstance(v, torch.Tensor) else v
                                if ("rank" in k or "entropy" in k) and val == 0: continue

                                if k not in val_probes_accum: val_probes_accum[k] = 0.0
                                val_probes_accum[k] += val
                            val_steps += 1

                if rank == 0:
                    eval_loss = total_loss / len(valid_loss_dataloader)
                    logger.info(f"Evaluation Loss (Epoch {epoch + 1}): {eval_loss}")

                    # [新增] 打印平均验证集 Probe
                    if val_steps > 0:
                        avg_val_probes = {k: round(v / val_steps, 4) for k, v in val_probes_accum.items()}
                        logger.info(f"Eval Probes (Epoch {epoch + 1}): {json.dumps(avg_val_probes)}")

        # val generation accuracy
        total_length = len(valid_gen_dataloader)

        pbar = tqdm(
            colour="blue", desc=f"Test Accuracy", total=total_length, dynamic_ncols=True
        )
        cor, cor_cot, total = (
            torch.tensor(0, device=rank),
            torch.tensor(0, device=rank),
            torch.tensor(0, device=rank),
        )

        with torch.no_grad():
            parallel_model.module.eval()
            for idx, batch in enumerate(valid_gen_dataloader):
                test_idx = batch["idx"][0]

                batch = {
                    k: v.to(rank)
                    for k, v in batch.items()
                    if v != None and k not in ["idx", "position_ids"]
                }

                assert len(batch["input_ids"]) == 1
                answer = answers_val[test_idx.cpu().item()]
                answer_cot = cot_val[test_idx.cpu().item()]
                question = question_val[test_idx.cpu().item()]

                total += 1

                outputs = parallel_model.module.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    synced_gpus=not configs.only_eval,
                )

                text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer_output = text_output.split("#")[-1].replace(",", "").strip()
                cot_output = (
                    ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()
                )

                if idx < 5 and rank == 0:
                    logger.info(f"[Example] Pred: {answer_output} | GT: {answer}")

                cor += answer_output == answer
                cor_cot += cot_output == answer_cot

                pbar.update(1)
                pbar.set_description(
                    f"Test accuracy: {round(float(cor.detach().float() / total.detach().float()), 2)}"
                )

            pbar.close()

        dist.all_reduce(cor_cot, op=dist.ReduceOp.SUM)
        dist.all_reduce(cor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

        cor_cot = cor_cot.item()
        cor = cor.item()
        total = total.item()
        if rank == 0:
            logger.info(f"Accuracy on validation set: {cor} / {total} = {cor / total}")
            logger.info(f"CoT match on validation set: {cor_cot} / {total} = {cor_cot / total}")
        sys.stdout.flush()

        if rank == 0:
            logger.info({"eval/acc": cor / total, "eval/cot_em": cor_cot / total})

        if configs.only_eval:
            break

        dist.barrier()
        if (
                cor / total > best_acc
                and configs.save_only_improve
                and not configs.debug
                and not configs.only_eval
        ):
            states = parallel_model.state_dict()

            if rank == 0:
                torch.save(states, os.path.join(save_dir, f"checkpoint_{epoch + 1}"))
                logger.info(f"New best model saved (Acc: {cor / total})")

            best_acc = cor / total

            dist.barrier()
            del states
            gc.collect()
            torch.cuda.empty_cache()

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
            print(f"Cleaned up stale lock file: {lock_path}")
        except:
            pass

    mp.spawn(worker, args=(world_size, args), nprocs=world_size)