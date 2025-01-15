def unlearn(
    input_path_to_unlearning_candidate_model: str,
    output_path_to_write_unlearned_model: str,
    path_to_forget_set: str,
    path_to_retain_set: str,
    tokenizer_path: str,
):
    """
    Memory-optimized unlearning implementation using LoRA and Sophia optimizer.
    Balanced for all types of content while maintaining reasonable GPU memory usage.
    """
    import gc
    import os
    import shutil
    from typing import Tuple

    import pandas as pd
    import torch
    import torch.nn as nn
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    NUM_EPOCHS = 4
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 4  
    LORA_RANK = 16 
    ACCUMULATION_STEPS = 8 
    MAX_LENGTH = 1024 
    DAMPING_FACTOR = 1e-3
    SOPHIA_RHO = 0.06
    SOPHIA_GAMMA = 1.2

    class Sophia(Optimizer):
        def __init__(
            self,
            params,
            lr=LEARNING_RATE,
            betas=(0.9, 0.999),
            rho=SOPHIA_RHO,
            gamma=SOPHIA_GAMMA,
            epsilon=1e-8,
            clipping_threshold=1.0,
        ):
            defaults = dict(
                lr=lr,
                betas=betas,
                rho=rho,
                gamma=gamma,
                epsilon=epsilon,
                clipping_threshold=clipping_threshold,
            )
            super(Sophia, self).__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad
                    state = self.state[p]

                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["hessian"] = torch.zeros_like(p)

                    exp_avg = state["exp_avg"]
                    hessian = state["hessian"]
                    beta1, beta2 = group["betas"]
                    state["step"] += 1

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                    if torch.rand(1) < group["rho"]:
                        with torch.enable_grad():
                            grad_squared = grad * grad
                        hessian.mul_(beta2).add_(grad_squared, alpha=1 - beta2)

                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]

                    denominator = torch.maximum(
                        group["gamma"] * hessian / bias_correction2,
                        torch.full_like(hessian, group["epsilon"]),
                    )

                    update = exp_avg / bias_correction1 / denominator
                    update.clamp_(
                        -group["clipping_threshold"], group["clipping_threshold"]
                    )
                    p.add_(update, alpha=-group["lr"])

            return loss

    class UnlearningDataset(Dataset):
        def __init__(self, file_path: str, tokenizer):
            self.df = pd.read_parquet(file_path)
            self.tokenizer = tokenizer

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

    def compute_woodfisher(
        model: nn.Module, dataloader: DataLoader, device: str
    ) -> dict:
        fisher_dict = {}
        model.train()

        for batch in tqdm(dataloader, desc="Computing WoodFisher"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name not in fisher_dict:
                        fisher_dict[name] = []
                    grad = param.grad.detach()
                    fisher_dict[name].append(grad.pow(2))

            model.zero_grad()
            torch.cuda.empty_cache() 

        # Average and compute inverse with dampening
        for name in fisher_dict:
            fisher_dict[name] = torch.stack(fisher_dict[name]).mean(0)
            fisher_dict[name] = 1.0 / (fisher_dict[name] + DAMPING_FACTOR)

        return fisher_dict

    # Setup device and directories
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(os.path.dirname(output_path_to_write_unlearned_model), exist_ok=True)
    checkpoint_dir = os.path.join(
        os.path.dirname(output_path_to_write_unlearned_model), "checkpoints"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        input_path_to_unlearning_candidate_model,
        device_map={"": device},
        torch_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ], 
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("Applying LoRA adaptation...")
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Create dataloaders
    forget_dataset = UnlearningDataset(path_to_forget_set, tokenizer)
    retain_dataset = UnlearningDataset(path_to_retain_set, tokenizer)

    forget_dataloader = DataLoader(
        forget_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    retain_dataloader = DataLoader(
        retain_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )

    # Phase 1: Influence-based Update
    print("Phase 1: Computing WoodFisher approximation...")
    woodfisher_inv = compute_woodfisher(model, retain_dataloader, device)

    print("Computing gradients on forget set...")
    forget_gradients = {}
    model.train()

    for batch in tqdm(forget_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                if name not in forget_gradients:
                    forget_gradients[name] = []
                forget_gradients[name].append(param.grad.detach().clone())

        model.zero_grad()
        torch.cuda.empty_cache()

    # Average gradients
    for name in forget_gradients:
        forget_gradients[name] = torch.stack(forget_gradients[name]).mean(0)

    # Apply influence-based update
    print("Applying influence-based parameter updates...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and name in woodfisher_inv:
                update = woodfisher_inv[name] * forget_gradients[name]
                param.data -= LEARNING_RATE * update

    del woodfisher_inv, forget_gradients
    torch.cuda.empty_cache()
    gc.collect()

    # Phase 2: Fine-tuning with Sophia
    print("Phase 2: Fine-tuning with Sophia optimizer...")
    optimizer = Sophia(model.parameters())

    best_loss = float("inf")
    best_checkpoint_path = None
    patience = 3
    patience_counter = 0

    try:
        for epoch in range(NUM_EPOCHS):
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for i, batch in enumerate(tqdm(forget_dataloader)):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / ACCUMULATION_STEPS
                loss.backward()

                if (i + 1) % ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * ACCUMULATION_STEPS

                # Save checkpoint every quarter epoch
                if (i + 1) % (len(forget_dataloader) // 4) == 0:
                    current_loss = epoch_loss / (i + 1)
                    checkpoint_path = os.path.join(
                        checkpoint_dir, f"checkpoint_epoch_{epoch + 1}_iter_{i + 1}.pt"
                    )
                    model.save_pretrained(checkpoint_path)

                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_checkpoint_path = checkpoint_path
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print("Early stopping triggered!")
                        raise StopIteration

                torch.cuda.empty_cache()

            avg_epoch_loss = epoch_loss / len(forget_dataloader)
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

    except (KeyboardInterrupt, StopIteration) as e:
        print(f"Training stopped: {str(e)}")

    finally:
        # Save final model
        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
            print(f"Saving best checkpoint as final model...")
            model = PeftModel.from_pretrained(base_model, best_checkpoint_path)
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(output_path_to_write_unlearned_model)
            tokenizer.save_pretrained(output_path_to_write_unlearned_model)
            print(f"Model saved to {output_path_to_write_unlearned_model}")

        # Cleanup
        print("Cleaning up...")
        for checkpoint in os.listdir(checkpoint_dir):
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
            if checkpoint_path != best_checkpoint_path:
                try:
                    shutil.rmtree(checkpoint_path)
                except Exception as e:
                    print(f"Error removing checkpoint {checkpoint_path}: {str(e)}")

    return model
