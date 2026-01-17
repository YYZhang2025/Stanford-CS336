import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse

# from cs336_alignment.dataset_utils.utils import convert_cot_to_think_answer
from cs336_alignment.drgrpo_grader import extract_answer


def convert_cot_to_think_answer(text: str, answer: str) -> str:
    """
    Convert a chain-of-thought style answer into the desired format by appending
    " </think> <answer> {answer} </answer>" at the end.
    Examples
    --------
    s = (
        the denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$.  therefore, the graph has $\boxed{2}$ vertical asymptotes.
        )
    answer = 2

    convert_cot_to_think_answer(s, answer)
    "the denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$.  therefore, the graph has $\boxed{2}$ vertical asymptotes. </think> <answer> 2 </answer>"
    """
    cot = text.rstrip()
    return f"{cot} </think> <answer> {str(answer)} </answer>"


# Regex: capture ints / floats / fractions; we will pick the LAST match as a fallback.
_NUM_RE = re.compile(r"(?<!\w)-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?(?!\w)")

# Math-Verify extraction config (same as you use in is_latex_equal)
_MV_CFG = (
    LatexExtractionConfig(boxed_match_priority=0),
    ExprExtractionConfig(),
)


def extract_final_answer_from_text(generated: str) -> Optional[str]:
    """
    Extract final answer string from a model-generated response.

    Priority:
      1) <answer>...</answer> (R1-zero format)
      2) \\boxed{...}
      3) math_verify parse() extraction from free-form text
      4) last numeric token fallback
    """
    if generated is None:
        return None

    s = generated.strip()

    # 1) R1-zero format: </think> <answer> ... </answer>
    if "<answer>" in s and "</answer>" in s:
        s = s.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()

    # 2) boxed
    if "\\boxed" in s:
        boxed = extract_answer(s)  # your existing function
        if boxed is not None:
            return boxed.strip()

    # 3) math-verify extraction (works well for "Therefore the answer is 18.")
    try:
        out = parse(
            s,
            extraction_config=_MV_CFG,
            fallback_mode="no_fallback",
            extraction_mode=["first_match"],
            parsing_timeout=1,
        )
        # Many versions return something like [sympy_obj, extracted_str]
        if out:
            if len(out) > 1 and isinstance(out[1], str) and out[1].strip():
                return out[1].strip()
            return str(out[0]).strip()
    except Exception:
        pass

    # 4) numeric fallback: take last number/fraction
    nums = _NUM_RE.findall(s)
    if nums:
        return nums[-1].replace(",", "").strip()

    return None


def collect_rows_jsonl(data_dir: str, filename: str = "train.json") -> List[Dict[str, Any]]:
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


def process_row(row: Dict[str, Any], prompt_template: str):
    problem = row["problem"]
    cot = row["solution"]

    if row["answer"] is None:
        answer = extract_final_answer_from_text(cot)
    else:
        answer = row["answer"]
    cot = convert_cot_to_think_answer(cot, answer)

    prompt = prompt_template.replace("{question}", problem)
    return prompt, str(cot).lower(), str(answer).lower() if answer is not None else None


def process_math(data_dir: str, prompt_template: str):
    train_prompts = []
    train_cots = []
    train_answers = []

    test_prompts = []
    test_cots = []
    test_answers = []

    train_rows = collect_rows_jsonl(data_dir, filename="train.json")
    test_rows = collect_rows_jsonl(data_dir, filename="test.json")

    for row in train_rows[:5]:
        prompt, cot, answer = process_row(row, prompt_template)
        train_prompts.append(prompt)
        train_cots.append(str(cot).lower())
        train_answers.append(str(answer).lower() if answer is not None else None)

    for row in test_rows[:10]:
        prompt, cot, answer = process_row(row, prompt_template)
        test_prompts.append(prompt)
        test_cots.append(str(cot).lower())
        test_answers.append(str(answer).lower() if answer is not None else None)

    return {
        "train": (train_prompts, train_cots, train_answers),
        "test": (test_prompts, test_cots, test_answers),
    }
