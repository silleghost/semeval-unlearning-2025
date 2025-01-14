def unlearn(
    input_path_to_unlearning_candidate_model: str,
    output_path_to_write_unlearned_model: str,
    path_to_forget_set: str,
    path_to_retain_set: str,
    tokenizer_path: str,
    num_epochs: int = 4,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    lora_rank: int = 32,
    damping_factor: float = 1e-3,
    sophia_rho: float = 0.06,
    sophia_gamma: float = 1.2,
    accumulation_steps: int = 4,
):
    """
    Unlearning implementation using LoRA and Sophia optimizer.
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

    class UnlearningDataset(Dataset):
        def __init__(self, file_path: str, tokenizer, max_length: int = 2048):
            self.df = pd.read_parquet(file_path)
            self.tokenizer = tokenizer
            self.max_length = max_length

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

    def compute_adaptive_damping(
        model, dataloader, device, initial_damping=1e-3, min_damping=1e-5
    ):
        """Compute adaptive damping factor based on gradient statistics."""
        grad_norms = []
        model.train()

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norms.append(total_norm**0.5)

            model.zero_grad()

        mean_norm = sum(grad_norms) / len(grad_norms)
        return max(min_damping, initial_damping * (mean_norm**0.5))

    def compute_woodfisher(
        model: nn.Module, dataloader: DataLoader, device: str, damping: float
    ) -> dict:
        fisher_dict = {}
        importance_weights = {}
        model.train()

        # Compute importance weights
        print("Computing importance weights...")
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                weight = torch.sigmoid(loss - loss.mean())

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if name not in importance_weights:
                            importance_weights[name] = []
                        importance_weights[name].append(weight)

        # Compute weighted Fisher
        print("Computing WoodFisher approximation...")
        for batch in tqdm(dataloader):
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
                    weight = importance_weights[name].pop(0)
                    fisher_dict[name].append(grad.pow(2) * weight)

            model.zero_grad()

        # Average and compute inverse with dampening
        for name in fisher_dict:
            fisher_dict[name] = torch.stack(fisher_dict[name]).mean(0)
            fisher_dict[name] = 1.0 / (fisher_dict[name] + damping)

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

    # Optimized LoRA configuration
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
        ],  # Added gate_proj
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Create PEFT model
    print("Applying LoRA adaptation...")
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Create dataloaders with dynamic batch sizing
    forget_dataset = UnlearningDataset(path_to_forget_set, tokenizer)
    retain_dataset = UnlearningDataset(path_to_retain_set, tokenizer)

    forget_dataloader = DataLoader(
        forget_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    retain_dataloader = DataLoader(
        retain_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Compute adaptive damping
    print("Computing adaptive damping factor...")
    adaptive_damping = compute_adaptive_damping(model, retain_dataloader, device)

    # Phase 1: Influence-based Update
    print("Phase 1: Applying influence-based update...")
    woodfisher_inv = compute_woodfisher(
        model, retain_dataloader, device, adaptive_damping
    )

    # Compute gradients on forget set
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

    # Average gradients
    for name in forget_gradients:
        forget_gradients[name] = torch.stack(forget_gradients[name]).mean(0)

    # Apply influence-based update
    print("Applying influence-based parameter updates...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and name in woodfisher_inv:
                update = woodfisher_inv[name] * forget_gradients[name]
                param.data -= learning_rate * update

    # Phase 2: Fine-tuning with Sophia
    print("Phase 2: Fine-tuning with Sophia optimizer...")

    # Initialize Sophia optimizer with layer-wise learning rate decay
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Apply higher learning rate to attention layers
        if any(module in name for module in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            lr_scale = 1.2
        else:
            lr_scale = 1.0

        param_groups.append({"params": [param], "lr": learning_rate * lr_scale})

    optimizer = Sophia(
        param_groups,
        rho=sophia_rho,
        gamma=sophia_gamma,
        epsilon=1e-8,
        clipping_threshold=1.0,
    )

    # Training loop with gradient accumulation
    best_forget_loss = float("inf")
    best_checkpoint_path = None
    patience = 3
    patience_counter = 0

    try:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
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
                loss = outputs.loss / accumulation_steps
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * accumulation_steps

                # Checkpoint saving logic
                if (i + 1) % (len(forget_dataloader) // 4) == 0:
                    current_loss = epoch_loss / (i + 1)

                    # Save checkpoint
                    checkpoint_path = os.path.join(
                        checkpoint_dir, f"checkpoint_epoch_{epoch + 1}_iter_{i + 1}.pt"
                    )
                    model.save_pretrained(checkpoint_path)

                    if current_loss < best_forget_loss:
                        best_forget_loss = current_loss
                        best_checkpoint_path = checkpoint_path
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Early stopping
                    if patience_counter >= patience:
                        print("Early stopping triggered!")
                        raise StopIteration

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

        # Cleanup checkpoints
        print("Cleaning up checkpoints...")
        for checkpoint in os.listdir(checkpoint_dir):
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
            if checkpoint_path != best_checkpoint_path:
                try:
                    shutil.rmtree(checkpoint_path)
                except Exception as e:
                    print(f"Error removing checkpoint {checkpoint_path}: {str(e)}")

    return model
