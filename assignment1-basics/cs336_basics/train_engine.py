from __future__ import annotations

import numpy as np
import torch
import wandb

from cs336_basics.config import TrainingConfig
from cs336_basics.generate import generate
from cs336_basics.loss import perplexity


@torch.no_grad()
def eval_model(
    model: torch.nn.Module,
    train_config: TrainingConfig,
):
    model.eval()

    eval_loss = 0.0
    eval_perplexity = 0.0

    with torch.no_grad():
        # Load evaluation dataset
        original_data = np.load(train_config.eval_data_path)

        for _ in range(train_config.num_eval_batches):
            inputs, targets = data_loading(
                x=original_data,
                batch_size=train_config.batch_size,
                context_length=model.config.max_seq_len,
                device=next(model.parameters()).device,
            )

            # Forward pass
            logits = model(inputs)
            loss = model.loss_fn(logits, targets)

            eval_loss += loss.item()
            eval_perplexity += perplexity(loss).item()

    eval_loss = torch.tensor(eval_loss / train_config.num_eval_batches)
    eval_perplexity = torch.tensor(eval_perplexity / train_config.num_eval_batches)

    # Logging
    if train_config.wandb_logging:
        wandb.log({"eval/loss": eval_loss.item()})
        wandb.log({"eval/perplexity": eval_perplexity.item()})

    model.train()


def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_config: TrainingConfig):
    # Load training dataset
    original_data = np.load(train_config.train_data_path)

    dataloader = data_loading(
        x=original_data,
        batch_size=train_config.batch_size,
        context_length=model.config.max_seq_len,
        device=next(model.parameters()).device,
    )

    # Training loop
    for step in range(train_config.num_training_steps):
        inputs, targets = dataloader

        # Forward pass
        logits = model(inputs)
        loss = model.loss_fn(logits, targets)

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if train_config.wandb_logging:
            wandb.log({"train/loss": loss.item(), "step": step})
            wandb.log({"train/perplexity": perplexity(loss).item(), "step": step})

        if train_config.log_interval > 0 and (step + 1) % train_config.log_interval == 0:
            eval_model(model, train_config)

            # Sample generation
            generated = generate(
                model,
                prompt_tokens=torch.tensor(
                    [[model.config.eos_token_id]], device=next(model.parameters()).device
                ),
                max_new_tokens=50,
                temperature=1.0,
                top_k=10,
            )
            generated_tokens = generated[0].cpu().numpy().tolist()
            if train_config.wandb_logging:
                wandb.log({"generated/tokens": generated_tokens, "step": step})
