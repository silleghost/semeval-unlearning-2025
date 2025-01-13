def unlearn(
    input_path_to_unlearning_candidate_model: str,
    output_path_to_write_unlearned_model: str,
    path_to_forget_set: str,
    path_to_retain_set: str,
    tokenizer_path: str,
    num_epochs: int = 3,
    learning_rate: float = 1e-5,
    batch_size: int = 8,
    lora_rank: int = 16,
    damping_factor: float = 1e-3,
    sophia_rho: float = 0.04,
    sophia_gamma: float = 1.0,
):
    """
    Implements influence-based unlearning using LoRA and Sophia optimizer.
    """
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

    # Sophia optimizer implementation
    class Sophia(Optimizer):
        def __init__(
            self,
            params,
            lr: float = 1e-4,
            betas: Tuple[float, float] = (0.9, 0.999),
            rho: float = 0.04,
            gamma: float = 1.0,
            epsilon: float = 1e-8,
            clipping_threshold: float = 1.0,
        ):
            if not 0.0 <= lr:
                raise ValueError(f"Invalid learning rate: {lr}")
            if not 0.0 <= epsilon:
                raise ValueError(f"Invalid epsilon value: {epsilon}")
            if not 0.0 <= betas[0] < 1.0:
                raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
            if not 0.0 <= betas[1] < 1.0:
                raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
            if not 0.0 <= rho <= 1.0:
                raise ValueError(f"Invalid rho value: {rho}")

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
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        state["hessian"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

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
                        min=-group["clipping_threshold"],
                        max=group["clipping_threshold"],
                    )
                    p.add_(update, alpha=-group["lr"])

            return loss

    # Internal Dataset class
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
                max_length=512,
                return_tensors="pt",
            )

            return {
                "input_ids": encodings["input_ids"].squeeze(),
                "attention_mask": encodings["attention_mask"].squeeze(),
                "labels": encodings["input_ids"].squeeze(),
            }

    def compute_gradient(model: nn.Module, dataloader: DataLoader, device: str) -> dict:
        """Compute gradients for influence-based update."""
        gradients = {}
        model.train()

        for batch in tqdm(dataloader, desc="Computing gradients"):
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
                    if name not in gradients:
                        gradients[name] = []
                    gradients[name].append(param.grad.detach().clone())

            model.zero_grad()

        # Average gradients
        for name in gradients:
            gradients[name] = torch.stack(gradients[name]).mean(0)

        return gradients

    def compute_woodfisher(
        model: nn.Module, dataloader: DataLoader, device: str
    ) -> dict:
        """Compute WoodFisher approximation for influence-based update."""
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

        # Average and compute inverse with dampening
        for name in fisher_dict:
            fisher_dict[name] = torch.stack(fisher_dict[name]).mean(0)
            fisher_dict[name] = 1.0 / (fisher_dict[name] + damping_factor)

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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        input_path_to_unlearning_candidate_model,
        device_map={"": device},
        torch_dtype=torch.float16,
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Create PEFT model
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Create dataloaders
    forget_dataset = UnlearningDataset(path_to_forget_set, tokenizer)
    retain_dataset = UnlearningDataset(path_to_retain_set, tokenizer)

    forget_dataloader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True)
    retain_dataloader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)

    # Phase 1: Influence-based Update
    print("Phase 1: Applying influence-based update...")

    # Compute WoodFisher approximation on retain set
    woodfisher_inv = compute_woodfisher(model, retain_dataloader, device)

    # Compute gradients on forget set
    forget_gradients = compute_gradient(model, forget_dataloader, device)

    # Apply influence-based update
    print("Applying influence-based parameter updates...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and name in woodfisher_inv:
                update = woodfisher_inv[name] * forget_gradients[name]
                param.data -= learning_rate * update

    # Phase 2: Fine-tuning with Sophia
    print("Phase 2: Fine-tuning with Sophia optimizer...")

    # Initialize Sophia optimizer with custom parameters
    optimizer = Sophia(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        rho=sophia_rho,
        gamma=sophia_gamma,
        epsilon=1e-8,
        clipping_threshold=1.0,
    )

    def compute_loss(model, batch):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs.loss

    # Training loop with checkpointing
    best_forget_loss = float("inf")
    best_checkpoint_path = None
    last_checkpoint_path = None

    def save_checkpoint(epoch, iteration, loss, is_best=False):
        checkpoint_name = f"checkpoint_epoch_{epoch}_iter_{iteration}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        model.save_pretrained(checkpoint_path)

        checkpoint = {
            "epoch": epoch,
            "iteration": iteration,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }
        torch.save(checkpoint, os.path.join(checkpoint_path, "training_state.pt"))
        return checkpoint_path

    # Training loop
    try:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            model.train()
            epoch_loss = 0.0

            for i, batch in enumerate(
                tqdm(forget_dataloader, desc=f"Unlearning epoch {epoch + 1}")
            ):

                def closure():
                    optimizer.zero_grad()
                    loss = compute_loss(model, batch)
                    loss.backward()
                    return loss

                # Sophia step with closure
                loss = optimizer.step(closure)
                epoch_loss += loss.item()

                if (i + 1) % (len(forget_dataloader) // 4) == 0:
                    current_loss = epoch_loss / (i + 1)
                    checkpoint_path = save_checkpoint(epoch + 1, i + 1, current_loss)
                    last_checkpoint_path = checkpoint_path

                    if current_loss < best_forget_loss:
                        best_forget_loss = current_loss
                        best_checkpoint_path = checkpoint_path

            avg_epoch_loss = epoch_loss / len(forget_dataloader)
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

            checkpoint_path = save_checkpoint(
                epoch + 1, len(forget_dataloader), avg_epoch_loss
            )
            last_checkpoint_path = checkpoint_path

            if avg_epoch_loss < best_forget_loss:
                best_forget_loss = avg_epoch_loss
                best_checkpoint_path = checkpoint_path

    except Exception as e:
        print(f"Training interrupted: {str(e)}")

    finally:
        # Save final model
        final_checkpoint_path = (
            best_checkpoint_path or last_checkpoint_path or checkpoint_dir
        )

        if final_checkpoint_path and os.path.exists(final_checkpoint_path):
            print(f"Saving best checkpoint as final model...")
            model = PeftModel.from_pretrained(base_model, final_checkpoint_path)
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(output_path_to_write_unlearned_model)
            tokenizer.save_pretrained(output_path_to_write_unlearned_model)
            print(f"Model saved to {output_path_to_write_unlearned_model}")

        # Cleanup checkpoints
        if best_checkpoint_path:
            for checkpoint in os.listdir(checkpoint_dir):
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
                if checkpoint_path != best_checkpoint_path:
                    try:
                        shutil.rmtree(checkpoint_path)
                    except Exception as e:
                        print(f"Error removing checkpoint {checkpoint_path}: {str(e)}")
