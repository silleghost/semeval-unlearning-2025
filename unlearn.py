import gc
import os
import shutil
from typing import Literal, Tuple

import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class Sophia(Optimizer):
    """Sophia optimizer implementation for unlearning process."""
    
    def __init__(
        self,
        params,
        lr=3e-5,
        betas=(0.9, 0.999),
        rho=0.08,
        gamma=1.15,
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
                update.clamp_(-group["clipping_threshold"], group["clipping_threshold"])
                p.add_(update, alpha=-group["lr"])

        return loss


class UnlearningDataset(Dataset):
    """Dataset for handling both forget and retain sets."""
    
    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer,
        dataset_type: Literal["forget", "retain"],
        max_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.max_length = max_length

        if os.path.isdir(file_path):
            file_name = f"{dataset_type}.parquet"
            file_path = os.path.join(file_path, file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_name} not found in directory {file_path}")

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
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze(),
        }


def compute_woodfisher(
    model: nn.Module,
    dataloader: DataLoader,
    damping_factor: float = 8e-4,
) -> dict:
    """Compute WoodFisher matrix inverse approximation."""
    fisher_dict = {}
    model.train()

    for batch in tqdm(dataloader, desc="Computing WoodFisher"):
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
                fisher_dict[name].append(param.grad.detach().pow(2))

        model.zero_grad()
        torch.cuda.empty_cache()

    # Average and compute inverse with dampening
    for name in fisher_dict:
        fisher_dict[name] = torch.stack(fisher_dict[name]).mean(0)
        fisher_dict[name] = 1.0 / (fisher_dict[name] + damping_factor)

    return fisher_dict


def unlearn(
    input_path_to_unlearning_candidate_model: str,
    output_path_to_write_unlearned_model: str,
    path_to_forget_set: str,
    path_to_retain_set: str,
    num_epochs: int = 5,
    learning_rate: float = 3e-5,
    batch_size: int = 1,
    lora_rank: int = 24,
    accumulation_steps: int = 6,
    max_length: int = 1024,
    damping_factor: float = 8e-4,
    sophia_rho: float = 0.08,
    sophia_gamma: float = 1.15,
):
    """
    Memory-optimized unlearning implementation using LoRA and Sophia optimizer.
    
    Args:
        input_path_to_unlearning_candidate_model: Path to the original model
        output_path_to_write_unlearned_model: Output path for unlearned model
        path_to_forget_set: Path to forget set directory or file
        path_to_retain_set: Path to retain set directory or file
        num_epochs: Number of fine-tuning epochs
        learning_rate: Base learning rate
        batch_size: Training batch size
        lora_rank: LoRA rank
        accumulation_steps: Gradient accumulation steps
        max_length: Maximum sequence length
        damping_factor: Damping factor for WoodFisher
        sophia_rho: Sophia rho parameter
        sophia_gamma: Sophia gamma parameter
    """
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    # Setup directories
    output_dir = os.path.dirname(output_path_to_write_unlearned_model)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(input_path_to_unlearning_candidate_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        input_path_to_unlearning_candidate_model,
        torch_dtype=torch.float32,
    )

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    accelerator.print(f"Trainable parameters: {model.print_trainable_parameters()}")

    # Prepare datasets and dataloaders
    def create_dataloader(path: str, dataset_type: Literal["forget", "retain"]):
        dataset = UnlearningDataset(
            file_path=path,
            tokenizer=tokenizer,
            dataset_type=dataset_type,
            max_length=max_length,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

    forget_dataloader = create_dataloader(path_to_forget_set, "forget")
    retain_dataloader = create_dataloader(path_to_retain_set, "retain")

    # Prepare components with accelerator
    optimizer = Sophia(
        model.parameters(),
        lr=learning_rate,
        rho=sophia_rho,
        gamma=sophia_gamma,
    )
    forget_dataloader, retain_dataloader, model, optimizer = accelerator.prepare(
        forget_dataloader, retain_dataloader, model, optimizer
    )

    # Phase 1: Influence-based Update
    accelerator.print("Phase 1: Computing WoodFisher approximation...")
    woodfisher_inv = compute_woodfisher(
        accelerator.unwrap_model(model),
        retain_dataloader,
        damping_factor=damping_factor,
    )

    accelerator.print("Computing gradients on forget set...")
    forget_gradients = {}
    model.train()

    for batch in tqdm(forget_dataloader, desc="Forget set gradients"):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in forget_gradients:
                    forget_gradients[name] = []
                forget_gradients[name].append(accelerator.gather(param.grad).detach().clone())

        model.zero_grad()
        torch.cuda.empty_cache()

    # Average gradients and apply update
    accelerator.print("Applying influence-based parameter updates...")
    with torch.no_grad():
        for name in forget_gradients:
            grad = torch.stack(forget_gradients[name]).mean(0)
            if name in woodfisher_inv:
                param = accelerator.unwrap_model(model).get_parameter(name)
                param.data -= learning_rate * woodfisher_inv[name] * grad

    del woodfisher_inv, forget_gradients
    torch.cuda.empty_cache()
    gc.collect()

    # Phase 2: Fine-tuning with Sophia
    accelerator.print("\nPhase 2: Fine-tuning with Sophia optimizer...")
    best_loss = float("inf")
    best_checkpoint_path = None
    patience = 3
    patience_counter = 0

    try:
        for epoch in range(num_epochs):
            accelerator.print(f"\nEpoch {epoch + 1}/{num_epochs}")
            model.train()
            total_loss = 0.0

            for step, batch in enumerate(tqdm(forget_dataloader, desc="Training")):
                outputs = model(**batch)
                loss = outputs.loss / accumulation_steps
                accelerator.backward(loss)

                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += accelerator.gather(loss).detach().mean().item()

                # Checkpoint every 25% of epoch
                if (step + 1) % (len(forget_dataloader) // 4) == 0:
                    current_loss = total_loss / (step + 1)
                    checkpoint_path = os.path.join(
                        checkpoint_dir, f"checkpoint_epoch_{epoch+1}_step_{step+1}"
                    )
                    accelerator.unwrap_model(model).save_pretrained(checkpoint_path)

                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_checkpoint_path = checkpoint_path
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        accelerator.print("Early stopping triggered!")
                        raise StopIteration

            avg_loss = total_loss / len(forget_dataloader)
            accelerator.print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    except (KeyboardInterrupt, StopIteration):
        accelerator.print("Training interrupted, saving best checkpoint...")

    finally:
        # Save final model
        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
            accelerator.print(f"\nSaving best model to {output_path_to_write_unlearned_model}")
            model = PeftModel.from_pretrained(base_model, best_checkpoint_path)
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(output_path_to_write_unlearned_model)
            tokenizer.save_pretrained(output_path_to_write_unlearned_model)

        # Cleanup checkpoints
        accelerator.print("\nCleaning up checkpoints...")
        for item in os.listdir(checkpoint_dir):
            item_path = os.path.join(checkpoint_dir, item)
            if item_path != best_checkpoint_path:
                shutil.rmtree(item_path, ignore_errors=True)

    return accelerator.unwrap_model(model)