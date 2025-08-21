import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Callable, List
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
from cs336_alignment.evaluate import extract_reference_answer, load_and_format_prompts, run_vllm
from cs336_alignment.sft_utils import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)
from cs336_alignment.utils import get_run_name, save_model_and_tokenizer


@dataclass
class TrainConfig:
    experiment_name: str = "sft-qwen2.5"
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    data_path: str = "./data/gsm8k/train.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    training_steps: int = 256
    mixed_precision_training: bool = True
    learning_rate: float = 2e-5
    train_device: str = "cuda:0"

    num_example: int = 128
    # For evaluation
    eval_device: str = "cuda:1"
    eval_interval_steps: int = 32


@dataclass
class EvaluateConfig:
    data_path: str = "./data/gsm8k/test.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1024


def sft_collate_fn(batch, tokenizer):
    # unpack batch
    prompts, cot, answers = zip(*batch)  # each is a tuple of strings
    prompts = list(prompts)
    cot = list(cot)
    answers = list(answers)

    # tokenize and prepare tensors
    batch_enc = tokenize_prompt_and_output(prompts, cot, tokenizer)

    return {**batch_enc, "answers": torch.stack(answers)}


class SFTDataset(Dataset):
    def __init__(self, train_prompts, train_cot, train_answers, tokenizer):
        self.train_prompts = train_prompts
        self.train_cot = train_cot
        self.train_answers = train_answers

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.train_prompts)

    def __getitem__(self, idx: int) -> tuple[str, str, str]:
        prompt = self.train_prompts[idx]
        cot = self.train_cot[idx]
        answer = self.train_answers[idx]

        return prompt, cot, answer


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


# def evaluate_vllm(
#     vllm_model: LLM,
#     reward_fn: Callable[[str, str], dict[str, float]],
#     prompts: List[str],
#     cot: List[str],
#     true_answers: List[str],
#     eval_sampling_params: SamplingParams,
# ):
#     responses = get_response(vllm_model, prompts, eval_sampling_params)
#     allinfo_dict_list = []
#     for response, true_answer, prompt in zip(responses, true_answers, prompts):
#         extracted_answer = extract_reference_answer(response)
#         reward_dict = reward_fn(response, true_answer)

#         info_dict: dict[str, Union[str, float]] = {
#             "prompt": prompt,
#             "response": response,
#             "true_answer": true_answer,
#             "extracted_answer": extracted_answer,
#             **reward_dict,
#         }

#         allinfo_dict_list.append(info_dict)

#     return allinfo_dict_list


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
):
    responses = run_vllm(vllm_model, prompts, eval_sampling_params)
    allinfo_dict_list = []
    for response, answer, prompt in zip(responses, answers, prompts):
        # extracted_answer = extract_reference_answer(response)
        reward_dict = reward_fn(response, answer)
        allinfo_dict_list.append(reward_dict)

    overview = {"correct": 0, "format_wrong": 0, "answer_wrong": 0, "count": 0}
    for reward in allinfo_dict_list:
        overview["count"] += 1
        if reward["reward"] == 1:
            overview["correct"] += 1
        elif reward["format_reward"] == 1:
            overview["answer_wrong"] += 1
        else:
            overview["format_wrong"] += 1

    return overview


def evaluate_sft_model(config: EvaluateConfig, vllm: LLM, eval_step: int):
    prompts, cot, answers = load_and_format_prompts(config.data_path, config.prompt_path)

    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
    )

    results = evaluate_vllm(vllm, r1_zero_reward_fn, prompts, answers, sampling_params)

    wandb.log(
        {
            "eval/correct": results["correct"],
            "eval/answer_wrong": results["answer_wrong"],
            "eval/format_wrong": results["format_wrong"],
            "eval_step": eval_step,
        }
    )


def train_sft_model(train_config: TrainConfig, eval_config: EvaluateConfig, dataset, vllm=None):
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

    # ---------------------
    # Load Model and Tokenizer
    # ---------------------
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=train_config.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    ).to(train_config.train_device)
    print(f"[train] Model {train_config.model_name} loaded on {train_config.train_device}")
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    print("[train] Tokenizer loaded")

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: sft_collate_fn(batch, tokenizer),
    )
    print(f"[train] Dataloader initialized with batch size {train_config.batch_size}")

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
    print("[train] Optimizer initialized")

    # ---------------------
    # Training Process
    # ---------------------
    cur_step = 0
    loss_accum = 0
    while True:
        for i, data in enumerate(dataloader):
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

            loss_accum += loss
            if (i + 1) % train_config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                print(f"[train] Step {cur_step} | Loss: {loss_accum:.4f}")

                wandb.log({"train/loss": loss_accum, "train_step": cur_step})
                loss_accum = 0

                cur_step += 1

            if (
                (i + 1) % train_config.gradient_accumulation_steps == 0
                and cur_step % train_config.eval_interval_steps == 0
            ):
                print(f"[train] Step {cur_step}: saving model at {train_config.experiment_name}")
                save_model_and_tokenizer(
                    model, tokenizer, f"{train_config.experiment_name}_{train_config.num_example}"
                )

                # Run evaluatoin
                print(f"[eval] at step {cur_step}")
                load_model_into_vllm_instance(model, vllm)
                evaluate_sft_model(eval_config, vllm, eval_step=cur_step)
                print(f"[eval] Evaluation completed for step {cur_step}")

            if cur_step >= train_config.training_steps:
                break

        if cur_step >= train_config.training_steps:
            break

    save_model_and_tokenizer(model, tokenizer, f"{train_config.experiment_name}_{train_config.num_example}")
    print(f"[train] Training finished at step {cur_step}")

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
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dotenv.load_dotenv()
    train_config = TrainConfig()
    eval_config = EvaluateConfig()

    # Login Wandb
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)

    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    prompts, cot, answers = load_and_format_prompts(train_config.data_path, train_config.prompt_path)

    vllm = init_vllm(
        model_id=model_name, device=train_config.eval_device, seed=1234, gpu_memory_utilization=0.85
    )

    for num_samples in [128, 256, 512, 1024, "all"]:
        train_config.num_example = num_samples if num_samples != "all" else len(prompts)
        # train_dataset = all_datasets[:num_samples] if num_samples != "all" else all_datasets
        train_prompts = prompts[:num_samples] if num_samples != "all" else prompts
        train_cot = cot[:num_samples] if num_samples != "all" else cot
        train_answers = answers[:num_samples] if num_samples != "all" else answers

        train_dataset = SFTDataset(train_prompts, train_cot, train_answers, tokenizer)
        train_sft_model(train_config, eval_config=eval_config, dataset=train_dataset, vllm=vllm)
        # train_sft_model(train_config, eval_config=eval_config, dataset=train_dataset)

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
