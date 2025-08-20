import argparse
import json
import re
from typing import Callable, List, Union

from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import extract_reference_answer


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
    texts = [output.outputs[0].text.strip() for output in result]
    return texts


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


def parse_args():
    parse_args = argparse.ArgumentParser(description="Evaluate VLLM model with prompts.")

    parse_args.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        required=True,
        help="Name of the VLLM model.",
    )
    parse_args.add_argument(
        "--data_path",
        type=str,
        default="./data/gsm8k/test.jsonl",
        required=True,
        help="Path to the reward function.",
    )
    parse_args.add_argument(
        "--prompt_path",
        type=str,
        default="./prompts/r1_zero.prompt",
        required=True,
        help="Path to the prompt template file.",
    )
    parse_args.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature for the model."
    )
    parse_args.add_argument(
        "--top_p", type=float, default=1.0, help="Top-p sampling parameter for the model."
    )
    parse_args.add_argument(
        "--max_tokens", type=int, default=1024, help="Maximum number of tokens to generate."
    )

    return parse_args.parse_args()


def main():
    args = parse_args()

    model_name = args.model_name
    data_path = args.data_path
    prompt_path = args.prompt_path

    vllm_model = LLM(model_name)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    prompts, answers = load_and_format_prompts(data_path, prompt_path)

    results = evaluate_vllm(vllm_model, r1_zero_reward_fn, prompts, answers, sampling_params)
    with open(f"evaluate_{model_name}_{data_path.split('/')[-1]}.jsonl", "w") as f:
        for i in results:
            json.dump(i, f)
            f.write("\n")


if __name__ == "__main__":
    main()
