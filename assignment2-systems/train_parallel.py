import os
import time
from dataclasses import dataclass
from typing import Tuple

import dotenv
import fire
import numpy as np
import torch
import torch.distributed as dist
from tqdm import trange

import wandb

# ---- your project imports ----
from cs336_basics.config import ModelConfig, TrainingConfig
from cs336_basics.generate import generate
from cs336_basics.loss import cross_entropy, perplexity
from cs336_basics.optim import cosine_annealing_lr, gradient_clip
from cs336_basics.tokenizer.tokenizer import load_tokenizer_from_dir
from cs336_basics.utils import clear_memory, get_ctx, print_color, save_checkpoint, seed_everything
from cs336_systems.parallel.ddp_bucket import DDPBucket
from cs336_systems.parallel.optim_shard import ShardedOptimizer
from cs336_systems.parallel.utils import cleanup_process_group


def init_distributed() -> Tuple[int, int, torch.device]:
    """Initialize torch.distributed (works with torchrun)."""
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        return rank, world_size, device

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return rank, world_size, device


def is_rank0() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


# 🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢
# Define Dataloader function, allow sample according different ranks
# 🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢
@dataclass
class BatchState:
    pos: int = 0
    world_size: int = 1


def get_batch_sequential_sharded(
    x_t: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
    state: BatchState,
    *,
    stride: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if stride is None:
        stride = context_length

    n = x_t.numel()
    max_start = n - context_length - 1
    if max_start < 0:
        raise ValueError(f"Sequence too short: n={n}, context_length={context_length}")

    last_start = state.pos + (batch_size - 1) * stride
    end = last_start + context_length + 1
    if end > n:
        state.pos = 0
        last_start = (batch_size - 1) * stride
        end = last_start + context_length + 1

    base = x_t[state.pos : end]  # 1D contiguous slice
    inputs = base.as_strided(size=(batch_size, context_length), stride=(stride, 1))
    targets = base[1:].as_strided(size=(batch_size, context_length), stride=(stride, 1))

    # shard across ranks
    state.pos += batch_size * stride * state.world_size

    inputs = inputs.to(device, non_blocking=(device.type == "cuda")).long()
    targets = targets.to(device, non_blocking=(device.type == "cuda")).long()
    return inputs, targets


# 🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴
# End Define Dataloader function, allow sample according different ranks
# 🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴


# 🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢
# Start Training and Evaluation functions
# 🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢
@torch.no_grad()
def eval_model(model: torch.nn.Module, train_config: TrainingConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    original_data = np.memmap(train_config.eval_data_path, dtype=np.uint16, mode="r+")
    x = torch.from_numpy(original_data)

    total_tokens = len(original_data)
    num_eval_batches = total_tokens // (train_config.batch_size * train_config.max_seq_len)
    num_eval_batches = max(num_eval_batches, 1)

    state = BatchState(pos=0, world_size=1)
    eval_loss = 0.0
    eval_ppl = 0.0

    for _ in trange(num_eval_batches):
        inputs, targets = get_batch_sequential_sharded(
            x_t=x,
            batch_size=train_config.batch_size,
            context_length=train_config.max_seq_len,
            device=get_device(),
            state=state,
        )

        logits, aux = model(inputs)
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        loss = cross_entropy(logits, targets)

        eval_loss += loss.item()
        eval_ppl += perplexity(loss).item()

    eval_loss = torch.tensor(eval_loss / num_eval_batches)
    eval_ppl = torch.tensor(eval_ppl / num_eval_batches)

    model.train()
    return eval_loss, eval_ppl


def train(model: torch.nn.Module, train_config: TrainingConfig):
    rank, world_size, device = init_distributed()
    rank0 = is_rank0()

    global_bs = train_config.batch_size
    assert global_bs % world_size == 0, f"batch_size={global_bs} must be divisible by world_size={world_size}"
    local_bs = global_bs // world_size

    # model -> device, wrap bucketed DDP
    model = model.to(device)
    ddp_model = DDPBucket(model, bucket_size_mb=getattr(train_config, "bucket_size_mb", 25.0))

    # sharded optimizer state
    optimizer = ShardedOptimizer(
        ddp_model.module.parameters(),
        optimizer_cls=torch.optim.AdamW,
        lr=train_config.max_lr,
        betas=train_config.betas,
        weight_decay=train_config.weight_decay,
    )

    # load tokenizer
    tokenizer = load_tokenizer_from_dir(train_config.dataset_dir)

    # load training dataset
    original_data = np.memmap(train_config.train_data_path, dtype=np.uint16, mode="r+")
    x = torch.from_numpy(original_data)

    best_eval_loss = float("inf")
    ctx = get_ctx(train_config.use_mixed_precision, device)

    # disjoint sequential traversal across ranks
    init_pos = rank * local_bs * train_config.max_seq_len
    state = BatchState(pos=init_pos, world_size=world_size)

    for step in range(train_config.num_steps):
        log_dict = {}

        inputs, targets = get_batch_sequential_sharded(
            x_t=x,
            batch_size=local_bs,
            context_length=train_config.max_seq_len,
            device=device,
            state=state,
        )

        # forward
        with ctx:
            logits, aux = ddp_model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets_ = targets.view(-1)
            loss = cross_entropy(logits, targets_)

            # MoE aux losses (if enabled)
            if getattr(model.module.config, "use_moe", False):
                z_loss_scaled = aux["z_loss_scaled"]
                moe_layers = aux["moe_layers"]
                loss = loss + (z_loss_scaled / moe_layers)

                lb_loss = aux["lb_loss_scaled"]
                loss = loss + (lb_loss / moe_layers)

        # backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # ✅ 关键：等 bucket all-reduce 完成 + 平均梯度（必须在 step 前）
        ddp_model.finish_gradient_synchronization()

        # ✅ clip 必须在 global grad sync 之后
        gradient_clip(ddp_model.module.parameters(), max_l2_norm=train_config.max_grad_norm)

        # lr schedule
        lr = cosine_annealing_lr(
            t=step,
            alpha_max=train_config.max_lr,
            alpha_min=train_config.min_lr,
            Tw=train_config.warmup_steps,
            Tc=train_config.num_steps - train_config.warmup_steps,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.step()
        clear_memory()

        # logging (rank0 only)
        if train_config.wandb_logging and rank0:
            log_dict["train/loss"] = loss.item()
            log_dict["train/perplexity"] = perplexity(loss).item()
            log_dict["train/lr"] = lr

        if rank0:
            print_color(
                f"[rank0] Step {step + 1}/{train_config.num_steps}, Loss: {loss.item():.4f}, LR: {lr:.6f}",
                "green",
            )

        # eval + ckpt (rank0 only)
        if train_config.eval_log_interval > 0 and (step + 1) % train_config.eval_log_interval == 0 and rank0:
            del inputs, targets, logits, loss
            clear_memory()

            print_color("Evaluating model...", "blue")
            eval_loss, eval_ppl = eval_model(ddp_model.module, train_config)

            if train_config.wandb_logging:
                log_dict["eval/loss"] = eval_loss.item()
                log_dict["eval/perplexity"] = eval_ppl.item()

            print_color(
                f"Eval Loss: {eval_loss.item():.4f}, Eval Perplexity: {eval_ppl.item():.4f}",
                "blue",
            )

            if eval_loss.item() < best_eval_loss:
                best_eval_loss = eval_loss.item()
                print_color(f"New best eval loss: {best_eval_loss:.4f}", "yellow")

                out_path = os.path.join(
                    train_config.save_checkpoint_dir,
                    train_config.model_name,
                    f"best_model_step_{step + 1}.pt",
                )
                save_checkpoint(
                    model=ddp_model.module,
                    optimizer=optimizer,
                    iteration=step + 1,
                    out=out_path,
                    verbose=True,
                )

        # sampling (rank0 only)
        if (
            train_config.sampling_log_interval > 0
            and (step + 1) % train_config.sampling_log_interval == 0
            and rank0
        ):
            generated_outputs = generate(
                model=ddp_model.module,
                prompt="Once upon a time",
                tokenizer=tokenizer,
                max_new_tokens=256,
                top_k=50,
                temperature=0.8,
            )
            generated_text = generated_outputs["generated_text"]
            print_color(f"Generated text at step {step + 1}:", "cyan")
            print("Once upon a time", end="")
            print_color(f"{generated_text}\n", "cyan")

        if train_config.wandb_logging and log_dict and rank0:
            wandb.log(log_dict, step=step + 1)

    cleanup_process_group()


def get_device() -> torch.device:
    rank = dist.get_rank()
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    else:
        return torch.device("cpu")


def build_model(model_config, device: torch.device):
    from cs336_basics.model import TransformerLM

    model = TransformerLM(model_config)
    model = model.to(get_device())
    model.train()
    return model


def main(
    train_config_json: str | None = "./configs/pytorch_flash_attn/train_config.json",
    model_config_json: str | None = "./configs/pytorch_flash_attn/model_config.json",
):
    # ---- dist setup ----
    rank, world_size, device = init_distributed()
    rank0 = is_rank0()

    # ---- Load configs ----
    train_config = TrainingConfig.from_json(train_config_json) if train_config_json else TrainingConfig()
    model_config = ModelConfig.from_json(model_config_json) if model_config_json else ModelConfig()

    # model_config.attention_type = "none_flash_attention"
    # model_config.max_seq_len = 128
    # train_config.debug_mode = True
    # train_config.num_steps = 100
    # train_config.batch_size = 16
    # train_config.wandb_logging = False

    train_config.max_seq_len = model_config.max_seq_len

    # 如果你希望 train_config.batch_size 表示 GLOBAL batch size（推荐）
    # 那么 local batch = global/world_size，需要你在 train() 里用 local_bs
    # 这里可以做一个强约束：
    if world_size > 1:
        assert train_config.batch_size % world_size == 0, (
            f"train_config.batch_size (global) must be divisible by world_size. "
            f"Got {train_config.batch_size} vs {world_size}"
        )

    # ---- Save configs (rank0 only) ----
    out_dir = os.path.join(train_config.save_checkpoint_dir, train_config.model_name)
    if rank0:
        os.makedirs(out_dir, exist_ok=True)
        model_config.to_json(os.path.join(out_dir, "model_config.json"))
        train_config.to_json(os.path.join(out_dir, "train_config.json"))

    # ---- WandB (rank0 only) ----
    dotenv.load_dotenv()
    wandb_api = os.getenv("WANDB_API_KEY")

    if train_config.wandb_logging and rank0:
        if wandb_api is None:
            raise ValueError("WANDB_API_KEY not found in environment variables.")
        import wandb

        wandb.login(key=wandb_api)
        wandb.init(
            project="cs336-basics-assignment2",
            name=f"{train_config.model_name}_batch{train_config.batch_size}_steps{train_config.num_steps}",
            config={"model_config": model_config.to_dict(), "train_config": train_config.to_dict()},
        )

    # ---- Seed ----
    # 建议每个 rank seed 不同但可复现：seed + rank
    seed_everything(train_config.seed + rank)

    # ---- Build model ----
    model = build_model(model_config, device=device)

    # ---- Wrap with bucketed DDP ----
    # 你可以把 bucket_size_mb 放进 train_config 里，没有就用默认值
    bucket_mb = float(getattr(train_config, "bucket_size_mb", 25.0))
    ddp_model = DDPBucket(model, bucket_size_mb=bucket_mb)

    # ---- Train ----
    if rank0:
        print_color("Starting training...", "blue")
        print_color(f"[info] world_size={world_size}, global_batch={train_config.batch_size}", "blue")
        print_color(f"[info] Total steps: {train_config.num_steps}", "blue")

    start_t = time.perf_counter()

    train(model=ddp_model, train_config=train_config)

    elapsed_s = time.perf_counter() - start_t

    # ---- finalize ----
    if rank0:
        print_color("Training completed.", "blue")
        print_color(f"Elapsed time: {elapsed_s:.2f}s", "blue")

        if train_config.wandb_logging:
            import wandb

            wandb.log(
                {
                    "time/elapsed_s": elapsed_s,
                    "time/elapsed_min": elapsed_s / 60.0,
                    "speed/steps_per_s": train_config.num_steps / max(elapsed_s, 1e-9),
                }
            )
            wandb.summary["time/elapsed_s"] = elapsed_s
            wandb.summary["speed/steps_per_s"] = train_config.num_steps / max(elapsed_s, 1e-9)
            wandb.finish()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
