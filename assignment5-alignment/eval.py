import csv
from pathlib import Path

from vllm import SamplingParams

from cs336_alignment.eval import evaluate_responses
from cs336_alignment.utils import get_device, load_dataset, print_color, print_rich_dict
from cs336_alignment.vllm_utils import init_vllm

PROMPT_TEMPLATE_PATH = "cs336_alignment/prompts/r1_zero.prompt"
MODEL_NAME = "models/Qwen2.5-Math-1.5B"

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "eval_overview.csv"

TRAIN_MATH_DATASET_PATHS = ["data/pre-processed/math/train.jsonl", "data/pre-processed/gsm8k/train.jsonl"]
TEST_MATH_DATASET_PATHS = ["data/pre-processed/math/test.jsonl", "data/pre-processed/gsm8k/test.jsonl"]


if __name__ == "__main__":
    vllm = init_vllm(
        model_id=MODEL_NAME,
        device=str(get_device(rank=1)),
        seed=42,
        gpu_memory_utilization=0.85,
    )
    sampling_params = SamplingParams(
        max_tokens=1024, temperature=1, top_p=1, stop=["</answer>"], include_stop_str_in_output=True
    )

    res = {}
    csv_rows = []
    for train_dataset_path, test_dataset_path in zip(TRAIN_MATH_DATASET_PATHS, TEST_MATH_DATASET_PATHS):
        (train_prompts, train_cots, train_answers) = load_dataset(
            train_dataset_path,
            prompt_template_path=PROMPT_TEMPLATE_PATH,
        )
        (test_prompts, test_cots, test_answers) = load_dataset(
            test_dataset_path,
            prompt_template_path=PROMPT_TEMPLATE_PATH,
        )

        print_color(f"Evaluating {train_dataset_path} Train Set...", color="cyan")
        train_overview = evaluate_responses(
            vllm=vllm,
            prompts=train_prompts,
            answers=train_answers,
            sampling_params=sampling_params,
        )
        print_rich_dict(train_overview)
        res[train_dataset_path] = train_overview
        csv_rows.append(
            {
                "split": "train",
                "dataset_path": train_dataset_path,
                **train_overview,
            }
        )

        print_color(f"Evaluating {test_dataset_path} Test Set...", color="cyan")
        test_overview = evaluate_responses(
            vllm=vllm,
            prompts=test_prompts,
            answers=test_answers,
            sampling_params=sampling_params,
        )
        print_rich_dict(test_overview)
        res[test_dataset_path] = test_overview
        csv_rows.append(
            {
                "split": "test",
                "dataset_path": test_dataset_path,
                **test_overview,
            }
        )

    if csv_rows:
        fieldnames = []
        for row in csv_rows:
            for k in row.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print_color(f"Saved evaluation overview to: {OUTPUT_CSV}", color="green")
