import os

import numpy as np
import torch
from tqdm import trange

import wandb
from cs336_basics.config import TrainingConfig
from cs336_basics.data import BatchState, data_loading, data_loading_sequential
from cs336_basics.generate import generate
from cs336_basics.loss import cross_entropy, perplexity
from cs336_basics.utils import clear_memory, get_ctx, print_color, save_checkpoint


@torch.no_grad()
def eval_model(
    model: torch.nn.Module,
    train_config: TrainingConfig,
):
    model.eval()

    eval_loss = 0.0
    eval_perplexity = 0.0
    # Load evaluation dataset
    original_data = np.memmap(
        train_config.eval_data_path,
        dtype=np.uint16,
        mode="r+",
    )
    total_tokens = len(original_data)
    num_eval_batches = total_tokens // (train_config.batch_size * model.config.max_seq_len)

    state = BatchState(pos=0)
    with torch.no_grad():
        for _ in trange(num_eval_batches):
            # inputs, targets = data_loading(
            #     x=original_data,
            #     batch_size=train_config.batch_size,
            #     context_length=model.config.max_seq_len,
            #     device=next(model.parameters()).device,
            # )
            inputs, targets = data_loading_sequential(
                x=original_data,
                batch_size=train_config.batch_size,
                context_length=model.config.max_seq_len,
                device=next(model.parameters()).device,
                state=state,
            )

            # Forward pass
            logits = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = cross_entropy(logits, targets)

            eval_loss += loss.item()
            eval_perplexity += perplexity(loss).item()

    eval_loss = torch.tensor(eval_loss / num_eval_batches)
    eval_perplexity = torch.tensor(eval_perplexity / num_eval_batches)

    # Logging
    if train_config.wandb_logging:
        wandb.log({"eval/loss": eval_loss.item()})
        wandb.log({"eval/perplexity": eval_perplexity.item()})

    model.train()

    return eval_loss


def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_config: TrainingConfig):
    # Load training dataset
    original_data = np.memmap(
        train_config.train_data_path,
        dtype=np.uint16,
        mode="r+",
    )

    # dataloader = data_loading(
    #     x=original_data,
    #     batch_size=train_config.batch_size,
    #     context_length=model.config.max_seq_len,
    #     device=train_config.device,
    # )

    best_eval_loss = float("inf")
    ctx = get_ctx(train_config.use_mixed_precision, train_config.device)

    # Training loop
    state = BatchState(pos=0)
    for step in range(train_config.num_steps):
        # inputs, targets = dataloader
        inputs, targets = data_loading_sequential(
            x=original_data,
            batch_size=train_config.batch_size,
            context_length=model.config.max_seq_len,
            device=train_config.device,
            state=state,
        )

        # Forward pass
        with ctx:
            logits = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = cross_entropy(logits, targets)

        # Backward pass and optimization step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Logging
        if train_config.wandb_logging:
            wandb.log({"train/loss": loss.item(), "step": step})
            wandb.log({"train/perplexity": perplexity(loss).item(), "step": step})

        print_color(f"Step {step + 1}/{train_config.num_steps}, Loss: {loss.item():.4f}", "green")

        if train_config.eval_log_interval > 0 and (step + 1) % train_config.eval_log_interval == 0:
            print_color("Evaluating model...", "blue")
            eval_loss = eval_model(model, train_config)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                print_color(f"New best eval loss: {best_eval_loss:.4f}", "yellow")
                out_dir = os.path.join(
                    train_config.save_checkpoint_dir,
                    train_config.model_name,
                )
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_path = os.path.join(
                    train_config.save_checkpoint_dir,
                    train_config.model_name,
                    f"best_model_step_{step + 1}.pt",
                )
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=step + 1,
                    out=out_path,
                    verbose=True,
                )

        # Sample generation
        if train_config.sampling_log_interval > 0 and (step + 1) % train_config.sampling_log_interval == 0:
            generated_outputs = generate(
                model=model,
                prompt="Once upon a time",
                tokenizer_dir=train_config.dataset_dir,
                max_new_tokens=256,
                top_k=10,
                temperature=1.0,
            )
            generated_text = generated_outputs["generated_text"]
            all_text = generated_outputs["all_text"]
            print_color(f"Generated text at step {step + 1}:", "cyan")
            print("Once upon a time", end="")
            print_color(f"{generated_text}\n", "cyan")

            if train_config.wandb_logging:
                wandb.log({"generated/tokens": all_text, "step": step})

        del inputs, targets, logits, loss
        clear_memory()
