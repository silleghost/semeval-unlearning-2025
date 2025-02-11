def unlearn(
    input_path_to_unlearning_candidate_model: str,
    output_path_to_write_unlearned_model: str,
    path_to_forget_set: str,
    path_to_retain_set: str,
):
    import gc
    import os
    import time
    from typing import Any, Dict, Literal

    import deepspeed
    import pandas as pd
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    # Configuration
    BATCH_SIZE = 1
    NUM_EPOCHS = 3
    BASE_LR = 1e-6
    LORA_RANK = 16
    ACCUMULATION_STEPS = 128
    MAX_LENGTH = 1024
    INITIAL_DAMPING = 1e-3
    MOMENTUM = 0.85
    GRAD_CLIP = 1.0

    # Layer-wise learning rates (increasing for later layers)
    LAYER_LR_FACTORS = [1.1**i for i in range(24)]

    deepspeed_config = {
        "train_batch_size": BATCH_SIZE * ACCUMULATION_STEPS * world_size,
        "train_micro_batch_size_per_gpu": BATCH_SIZE,
        "gradient_accumulation_steps": ACCUMULATION_STEPS,
        "optimizer": {
            "type": "Adam",
            "params": {"weight_decay": 0.01, "betas": (0.9, 0.95), "torch_adam": True},
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "contiguous_gradients": True,
            "overlap_comm": True,
        },
        "gradient_clipping": GRAD_CLIP,
        "fp16": {"enabled": True},
    }

    class UnlearningDataset(Dataset):
        def __init__(
            self, file_path: str, tokenizer, dataset_type: Literal["forget", "retain"]
        ):
            self.tokenizer = tokenizer
            self.dataset_type = dataset_type

            if os.path.isdir(file_path):
                file_name = f"{dataset_type}.parquet"
                file_path = os.path.join(file_path, file_name)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"{file_name} not found in directory {file_path}"
                    )

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            self.df = pd.read_parquet(file_path)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            full_text = f"{row['input']} {self.tokenizer.sep_token} {row['output']}"
            encodings = self.tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            return {
                "input_ids": encodings["input_ids"].squeeze(),
                "attention_mask": encodings["attention_mask"].squeeze(),
                "labels": encodings["input_ids"].squeeze(),
            }

    def block_woodfisher(model, dataloader):
        """Block-wise Fisher approximation with adaptive damping"""
        fisher = {}
        model.train()
        grad_norms = []

        # First pass: collect gradient norms
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            model.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            model.backward(loss)

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_norms.append(param.grad.norm().item())

        # Adaptive damping based on gradient statistics
        damping = max(INITIAL_DAMPING, torch.tensor(grad_norms).mean().item() / 100)

        # Second pass: compute block-wise Fisher
        for batch in tqdm(dataloader, desc="Block Fisher"):
            batch = {k: v.to(device) for k, v in batch.items()}
            model.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            model.backward(loss)

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad = param.grad.detach().float()

                        # Block-wise outer product for attention layers
                        if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                            block = grad.view(-1, LORA_RANK)
                            fisher_block = block.T @ block
                        else:
                            fisher_block = grad.pow(2)

                        if name not in fisher:
                            fisher[name] = fisher_block.clone()
                        else:
                            fisher[name] += fisher_block

        # Average and invert with damping
        for name in fisher:
            fisher[name] = torch.inverse(
                fisher[name] / len(dataloader)
                + damping * torch.eye(fisher[name].size(-1), device=device)
            )
        return fisher

    def momentum_update(model, gradients, fisher, momentum_buffer):
        """Apply momentum-based influence update with clipping"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in fisher and name in gradients:
                    update = fisher[name] @ gradients[name]

                    # Gradient clipping
                    update_norm = torch.norm(update)
                    if update_norm > GRAD_CLIP:
                        update = update * GRAD_CLIP / update_norm

                    # Momentum integration
                    if name not in momentum_buffer:
                        momentum_buffer[name] = update.clone()
                    else:
                        momentum_buffer[name] = (
                            MOMENTUM * momentum_buffer[name] + update
                        )

                    param.data -= BASE_LR * momentum_buffer[name]

    tokenizer = AutoTokenizer.from_pretrained(input_path_to_unlearning_candidate_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        input_path_to_unlearning_candidate_model,
        torch_dtype=torch.float16,
        device_map={"": device},
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)

    # Assign layer-wise learning rates
    params_grouped = {}
    for layer_idx, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            lr_factor = LAYER_LR_FACTORS[layer_idx % len(LAYER_LR_FACTORS)]
            params_grouped.setdefault(lr_factor, []).append(param)

    optimizer_params = [
        {"params": params, "lr": BASE_LR * lr_factor}
        for lr_factor, params in params_grouped.items()
    ]

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=deepspeed_config,
        model_parameters=optimizer_params,
    )

    # Prepare data
    forget_loader = DataLoader(
        UnlearningDataset(path_to_forget_set, tokenizer, "forget"),
        batch_size=BATCH_SIZE,
        sampler=DistributedSampler(UnlearningDataset(path_to_forget_set, tokenizer, "forget")),
    )
    retain_loader = DataLoader(
        UnlearningDataset(path_to_retain_set, tokenizer, "retain"),
        batch_size=BATCH_SIZE,
        sampler=DistributedSampler(UnlearningDataset(path_to_retain_set, tokenizer, "retain")),
    )

    # Compute influence components
    fisher = block_woodfisher(model_engine, retain_loader)
    momentum_buffer = {}

    # Main unlearning loop
    try:
        for epoch in range(NUM_EPOCHS):
            model_engine.train()
            epoch_gradients = {
                n: torch.zeros_like(p, device=device)
                for n, p in model_engine.named_parameters()
                if p.requires_grad
            }

            for batch_idx, batch in enumerate(tqdm(forget_loader)):
                batch = {k: v.to(device) for k, v in batch.items()}
                model_engine.zero_grad()

                outputs = model_engine(**batch)
                loss = outputs.loss / ACCUMULATION_STEPS
                model_engine.backward(loss)

                # Accumulate gradients
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    for name, param in model_engine.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            epoch_gradients[name] += param.grad.detach()
                    model_engine.zero_grad()

            # Apply momentum update
            momentum_update(model_engine, epoch_gradients, fisher, momentum_buffer)

    finally:
        # Merge and save full model
        if local_rank == 0:
            peft_model = model_engine.module
            merged_model = peft_model.merge_and_unload()
            merged_model.save_pretrained(output_path_to_write_unlearned_model)
            tokenizer.save_pretrained(output_path_to_write_unlearned_model)
            print(f"Saved full model to {output_path_to_write_unlearned_model}")

        torch.distributed.destroy_process_group()
        torch.cuda.empty_cache()

