import argparse
import json
import re
from typing import Callable, List, Union

from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import extract_reference_answer, safe_slug


def load_and_format_prompts(data_path: str, prompt_path: str):
    with open(prompt_path, "r") as file:
        prompt = file.read()
    prompts = []
    answers = []
    with open(data_path, "r") as file:
        for line in file:
            data = json.loads(line)
            prompts.append(prompt.format(question=data["question"]))
            answers.append(data["answer"])

    return prompts, answers


def run_vllm(vllm_model, prompts, sampling_params) -> List[str]:
    result = vllm_model.generate(prompts, sampling_params)
    outputs = [output.outputs[0].text.strip() for output in result]
    return outputs


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
        extracted_answer = extract_reference_answer(answer)
        reward_dict = reward_fn(response, extracted_answer)

        info_dict: dict[str, Union[str, float]] = {
            **reward_dict,
            "response": response,
            "answer": answer,
            "prompt": prompt,
            "extracted_answer": extracted_answer,
        }

        allinfo_dict_list.append(info_dict)

    return allinfo_dict_list


import os
from pathlib import Path

import fire


def main(
    *,
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    data_path: str = "./data/gsm8k/test.jsonl",
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt",
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
):
    vllm_model = LLM(model_name)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    prompts, answers = load_and_format_prompts(data_path, prompt_path)

    results = evaluate_vllm(vllm_model, r1_zero_reward_fn, prompts, answers, sampling_params)

    model_tag = safe_slug(model_name)
    data_stem = Path(data_path).stem
    out_dir = Path("evaluations")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"evaluate_{model_tag}_{data_stem}.jsonl"

    with open(out_file, "w", encoding="utf-8") as f:
        for i in results:
            json.dump(i, f)
            f.write("\n")

    print(f"Wrote {out_file}")


if __name__ == "__main__":
    fire.Fire(main)
