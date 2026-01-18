import json
from pathlib import Path
from typing import Any, Dict, List

import regex as re

from cs336_alignment.utils import wrap_cot_with_answer

""" 
{"question": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", 
"answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer\u2019s market.\n#### 18"}
"""


def extract_gsm8k_answer(answer: str) -> str:
    ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"


def parse_gsm8k_model_output(response: str) -> str | None:
    """
    Extract the final answer from a GSM8K-style response.
    Looks for the pattern "#### <answer>" at the end of the response.
    """
    from cs336_alignment.dataset_utils.math import extract_answer

    model_answer = response.split("<answer>")[-1].replace("</answer>", "")
    if "\\boxed" in model_answer:
        model_answer = extract_answer(model_answer)

    return model_answer.strip() if model_answer else None


def collect_rows(data_dir: str, filename: str = "train.jsonl") -> List[Dict[str, Any]]:
    p = Path(data_dir) / filename
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")

    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def process_row(row: Dict[str, Any]):
    problem = row["question"]
    cot = row["answer"]
    clean_cot = re.sub(r"\s*\n####\s*-?\d+(?:\.\d+)?\s*$", "", cot)
    answer = extract_gsm8k_answer(row["answer"])

    clean_cot = wrap_cot_with_answer(clean_cot, answer)

    return problem, str(clean_cot), str(answer).lower() if answer is not None else None
