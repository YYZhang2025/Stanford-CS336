import multiprocessing as mp
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import List, Optional
from unittest.mock import patch

import dotenv
import fire
import torch
import wandb
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.evaluate import evaluate_vllm, load_and_format_prompts
from cs336_alignment.sft_utils import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)
from cs336_alignment.utils import get_run_name, load_json_to_list


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    data_path: str = "./data/gsm8k/train.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"
    batch_size: int = 2
    num_example: int = 128
    training_steps: int = 256
    mixed_precision_training: bool = True
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    train_device: str = "cuda:0"
    eval_device: str = "cuda:1"
    eval_interval_steps: int = 10
    ckpt_dir: str = "./checkpoints"


@dataclass
class EvaluateConfig:
    data_path: str = "./data/gsm8k/test.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1024


from torch.utils.data import DataLoader


def sft_collate_fn(batch, tokenizer):
    # unpack batch
    prompts, answers = zip(*batch)  # each is a tuple of strings
    prompts = list(prompts)
    answers = list(answers)

    # tokenize and prepare tensors
    batch_enc = tokenize_prompt_and_output(prompts, answers, tokenizer)

    return batch_enc


class SFTDataset(Dataset):
    def __init__(self, train_prompts, train_answers, tokenizer):
        self.train_prompts = train_prompts
        self.train_answers = train_answers

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.train_prompts)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        prompt = self.train_prompts[idx]
        answer = self.train_answers[idx]

        return prompt, answer


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_model_into_vllm_instance(model: PreTrainedModel, llm: LLM):
    state_dict = model.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def load_state_dict_into_vllm(state_dict: dict, llm: LLM):
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_sft_model(config: EvaluateConfig, vllm: LLM, eval_step: int):
    prompts, answers = load_and_format_prompts(config.data_path, config.prompt_path)

    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
    )

    results = evaluate_vllm(vllm, r1_zero_reward_fn, prompts, answers, sampling_params)

    wandb.log(
        {
            "eval/loss": results["loss"],
            "eval/accuracy": results["accuracy"],
            "eval_step": eval_step,
        }
    )


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _atomic_save_checkpoint(model: torch.nn.Module, ckpt_dir: str, step: int) -> str:
    _ensure_dir(ckpt_dir)
    tmp = os.path.join(ckpt_dir, f"step_{step}.pt.tmp")
    final = os.path.join(ckpt_dir, f"step_{step}.pt")
    torch.save({"step": step, "state_dict": model.state_dict()}, tmp)
    os.replace(tmp, final)  # atomic rename
    return final


def _latest_ckpt(ckpt_dir: str, last_step: int) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = []
    for n in os.listdir(ckpt_dir):
        if n.startswith("step_") and n.endswith(".pt"):
            try:
                s = int(n.split("_")[1].split(".")[0])
                if s > last_step:
                    candidates.append((s, os.path.join(ckpt_dir, n)))
            except Exception:
                continue
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def eval_worker(
    eval_cfg: EvaluateConfig,
    model_name: str,
    data_path: str,
    prompt_path: str,
    device: str,
    ckpt_dir: str,
    run_id: str,
):
    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
        id=run_id,
        resume="allow",
        reinit=True,
    )

    # vLLM on eval GPU
    vllm = init_vllm(model_id=model_name, device=device, seed=1234, gpu_memory_utilization=0.85)

    # local EvaluateConfig with provided paths
    local_eval_cfg = EvaluateConfig(
        temperature=eval_cfg.temperature,
        top_p=eval_cfg.top_p,
        max_tokens=eval_cfg.max_tokens,
    )

    # override paths used by evaluate_sft_model via monkeypatching attributes
    local_eval_cfg.data_path = data_path
    local_eval_cfg.prompt_path = prompt_path

    last_step = -1
    while True:
        ckpt_path = _latest_ckpt(ckpt_dir, last_step)
        if ckpt_path is None:
            time.sleep(2.0)
            continue

        payload = torch.load(ckpt_path, map_location="cpu")
        step = int(payload.get("step", last_step))
        state_dict = payload["state_dict"]
        load_state_dict_into_vllm(state_dict, vllm)
        evaluate_sft_model(local_eval_cfg, vllm, eval_step=step)
        last_step = step

        # small sleep to avoid hot loop
        time.sleep(2.0)


def train_sft_model(train_config: TrainConfig, eval_config: EvaluateConfig, dataset):
    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project="cs336-alignment-sft",
        config={
            "train": asdict(train_config),
            "eval": asdict(eval_config),
        },
        name=get_run_name("sft", train_config),
        reinit=True,
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    run = wandb.run
    run_id = run.id

    # kick off async evaluator
    eval_proc = mp.Process(
        target=eval_worker,
        args=(
            eval_config,
            train_config.model_name,
            train_config.data_path,
            train_config.prompt_path,
            train_config.eval_device,
            train_config.ckpt_dir,
            run_id,
        ),
        daemon=True,
    )
    eval_proc.start()

    # ---------------------
    # Load Model and Tokenizer
    # ---------------------
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=train_config.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    ).to(train_config.train_device)
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: sft_collate_fn(batch, tokenizer),
    )

    ctx = (
        nullcontext()
        if train_config.mixed_precision_training
        else torch.autocast("cuda", dtype=torch.bfloat16)
    )

    # ---------------------
    # Optimizer
    # ---------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # ---------------------
    # Training Process
    # ---------------------
    cur_step = 0
    while True:
        for data in dataloader:
            #  {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}
            input_ids = data["input_ids"].to(train_config.train_device)
            labels = data["labels"].to(train_config.train_device)
            response_mask = data["response_mask"].to(train_config.train_device)

            with ctx:
                log_prob = get_response_log_probs(model=model, input_ids=input_ids, labels=labels)
                log_prob = log_prob["log_probs"]
                loss, info = sft_microbatch_train_step(
                    log_prob, response_mask, train_config.gradient_accumulation_steps
                )

            wandb.log({"train/loss": loss.item(), "train_step": cur_step})
            cur_step += 1

            if cur_step % train_config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Async eval: checkpoint every N steps; eval_worker will pick it up
            if cur_step % train_config.eval_interval_steps == 0:
                _atomic_save_checkpoint(model, train_config.ckpt_dir, cur_step)

            if cur_step >= train_config.training_steps:
                break

        if cur_step >= train_config.training_steps:
            break

    _atomic_save_checkpoint(model, train_config.ckpt_dir, cur_step)
    wandb.finish()
    return model


def main(
    *,
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    data_path: str = "./data/gsm8k/train.jsonl",
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt",
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
):
    dotenv.load_dotenv()
    train_config = TrainConfig()

    # Login Wandb
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)

    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    prompts, answers = load_and_format_prompts(train_config.data_path, train_config.prompt_path)

    os.makedirs(train_config.ckpt_dir, exist_ok=True)

    for num_samples in [128, 256, 512, 1024, "all"]:
        train_config.num_example = num_samples if num_samples != "all" else len(prompts)
        # train_dataset = all_datasets[:num_samples] if num_samples != "all" else all_datasets
        train_prompts = prompts[:num_samples] if num_samples != "all" else prompts
        train_answers = answers[:num_samples] if num_samples != "all" else answers

        train_dataset = SFTDataset(train_prompts, train_answers, tokenizer)
        train_sft_model(train_config, eval_config=EvaluateConfig(), dataset=train_dataset)

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
