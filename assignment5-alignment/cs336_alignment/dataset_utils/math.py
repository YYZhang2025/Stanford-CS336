import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse

from cs336_alignment.drgrpo_grader import extract_answer
from cs336_alignment.utils import wrap_cot_with_answer

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


def collect_rows(data_dir: str, filename: str = "train.json") -> List[Dict[str, Any]]:
    p = Path(data_dir) / filename
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")

    rows: List[Dict[str, Any]] = []
    df = pd.read_parquet(p)
    print(df.iloc[0][0])
    for _, row in df.iterrows():
        rows.append(row.to_dict())
    return rows


def process_row(row: Dict[str, Any]):
    problem = row["problem"]
    cot = row["solution"]

    if row["answer"] is None:
        answer = extract_final_answer_from_text(cot)
    else:
        answer = row["answer"]
        cot = wrap_cot_with_answer(cot, answer)

    return problem, str(cot), str(answer).lower() if answer is not None else None
