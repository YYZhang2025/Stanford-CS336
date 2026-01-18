import json

from torch.utils.data import Dataset
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import extract_answer, r1_zero_reward_fn
from cs336_alignment.eval import evaluate_responses
from cs336_alignment.utils import (
    get_device,
    print_color,
)
from cs336_alignment.vllm_utils import generate_responses, init_vllm


def extract_reference_answer(response: str) -> str:
    model_answer = response.split("<answer>")[-1].replace("</answer>", "")
    if "\\boxed" in model_answer:
        model_answer = extract_answer(model_answer)

    return model_answer


def evaluate_responses(
    vllm: LLM,
    prompts: list[str],
    answers: list[str],
    sampling_params,
):
    responses = generate_responses(vllm, prompts, sampling_params)
    allinfo_list = []
    for response, answer, prompt in zip(responses, answers, prompts):
        reward_dict = r1_zero_reward_fn(response, ground_truth=answer)
        allinfo_list.append(reward_dict)

    # Gather statistics
    overview = {
        "total": len(responses),
        "correct": 0,
        "format_wrong": 0,
        "answer_wrong": 0,
    }

    for reward_dict in allinfo_list:
        if reward_dict["reward"] == 1:
            overview["correct"] += 1
        elif not reward_dict["format_reward"]:
            overview["format_wrong"] += 1
        else:
            overview["answer_wrong"] += 1

    return overview


class SFTDataset(Dataset):
    def __init__(self, questions: list[str], cots: list[str], answers: list[str], prompt_template_path: str):
        self.questions = questions
        self.cots = cots
        self.answers = answers

        with open(prompt_template_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        self.prompts = [self.prompt_template.format(question=q) for q in self.questions]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        cot = self.cots[idx]
        answer = self.answers[idx]

        return prompt, cot, answer

    @classmethod
    def load_from_disk(cls, path: str, prompt_template_path: str):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        questions = []
        cots = []
        answers = []
        for row in rows:
            questions.append(row["question"])
            cots.append(row["cot"])
            answers.append(row["answer"])

        return cls(questions, cots, answers, prompt_template_path=prompt_template_path)


if __name__ == "__main__":
    TRAIN_MATH_DATASET_PATH = "data/pre-processed/math/train.jsonl"
    TEST_MATH_DATASET_PATH = "data/pre-processed/math/test.jsonl"
    PROMPT_TEMPLATE_PATH = "cs336_alignment/prompts/r1_zero.prompt"
    MODEL_NAME = "models/Qwen2.5-Math-1.5B"
    train_dataset = SFTDataset.load_from_disk(
        TRAIN_MATH_DATASET_PATH,
        prompt_template_path=PROMPT_TEMPLATE_PATH,
    )
    test_dataset = SFTDataset.load_from_disk(
        TEST_MATH_DATASET_PATH,
        prompt_template_path=PROMPT_TEMPLATE_PATH,
    )

    train_prompts = train_dataset.prompts
    train_answers = train_dataset.answers
    test_prompts = test_dataset.prompts
    test_answers = test_dataset.answers

    vllm = init_vllm(
        model_id=MODEL_NAME,
        device=get_device(rank=1),
        seed=42,
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(
        max_tokens=1024, temperature=1, top_p=1, stop=["</answer>"], include_stop_str_in_output=True
    )

    print_color("Evaluating Training Set...", color="cyan")
    train_overview = evaluate_responses(
        vllm=vllm,
        prompts=train_prompts,
        answers=train_answers,
        sampling_params=sampling_params,
    )
    print_color(f"Training Set Evaluation: {train_overview}", color="green")
    train_overview["accuracy"] = train_overview["correct"] / train_overview["total"]

    print_color(
        f"SFT Evaluation Results - Total: {train_overview['total']}, Correct: {train_overview['correct']}, "
        f"Format Wrong: {train_overview['format_wrong']}, Answer Wrong: {train_overview['answer_wrong']}, "
        f"Accuracy: {train_overview['accuracy']:.4f}",
        color="magenta",
    )
    print_color("Evaluating Test Set...", color="cyan")
    test_overview = evaluate_responses(
        vllm=vllm,
        prompts=test_prompts,
        answers=test_answers,
        sampling_params=sampling_params,
    )

    print_color(f"Test Set Evaluation: {test_overview}", color="green")
    test_overview["accuracy"] = test_overview["correct"] / test_overview["total"]
    print_color(
        f"SFT Evaluation Results - Total: {test_overview['total']}, Correct: {test_overview['correct']}, "
        f"Format Wrong: {test_overview['format_wrong']}, Answer Wrong: {test_overview['answer_wrong']}, "
        f"Accuracy: {test_overview['accuracy']:.4f}",
        color="magenta",
    )
