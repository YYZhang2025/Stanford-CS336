import json
import os
import re
from typing import Any

import pandas as pd
import regex as re

MMLU_TEMPLATE = """ {question}\nA. {option_A}\nB. {option_B}\nC. {option_C}\nD. {option_D}"""


def collect_rows(data_dir: str) -> list[dict]:
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    mmlu_example_list = []
    # for filename in csv_files:
    #     p = os.path.join(data_dir, filename)
    #     df = pd.read_csv(p, header=None)
    #     for _, row in df.iterrows():
    #         question = MMLU_TEMPLATE.format(
    #             question=row[0],
    #             option_A=row[1],
    #             option_B=row[2],
    #             option_C=row[3],
    #             option_D=row[4],
    #         )
    #         answer = row[5].strip()
    #         question_answers.append({"question": question, "answer": answer})
    # return question_answers
    for filename in csv_files:
        subject = filename.split(".")[0]
        p = os.path.join(data_dir, filename)
        df = pd.read_csv(p)
        for _, row in df.iterrows():
            question = row.iloc[0]
            options = [row.iloc[1], row.iloc[2], row.iloc[3], row.iloc[4]]
            answer = row.iloc[5]

            # question_formatted = MMLU_TEMPLATE.format(
            #     question=question,
            #     option_A=options[0],
            #     option_B=options[1],
            #     option_C=options[2],
            #     option_D=options[3],
            # )

            mmlu_example_list.append(
                {
                    "subject": subject,
                    "question": question,
                    "options": options,
                    "answer": answer,
                }
            )
    return mmlu_example_list


def parse_mmlu_model_output(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> tuple[str | None, bool]:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.
    """

    # Normalize a bit
    text = model_output.replace("\u200b", " ")

    pred: str | None = None

    # Prefer explicit "answer" patterns.
    keyword_patterns = [
        r"(?i)\b(?:final\s*answer|final|answer|correct\s*answer|correct\s*option|option|choice|selected)\b\s*[:\-\)]*\s*\(?\s*([ABCD])\s*\)?\b",
        r"(?i)\b(?:is|=)\s*\(?\s*([ABCD])\s*\)?\b\s*[\.!]?$",
    ]
    for pat in keyword_patterns:
        m = re.search(pat, text)
        if m:
            pred = m.group(1).upper()
            break

    # 1) Single letter on the last non-empty line.
    if pred is None:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            last = lines[-1]
            m = re.fullmatch(r"(?i)\(?\s*([ABCD])\s*\)?\.?", last)
            if m:
                pred = m.group(1).upper()

    # 2) Letter at the very end of the output.
    if pred is None:
        m = re.search(r"(?i)(?:^|\s|\(|\[|\")([ABCD])(?:\s|\)|\]|\"|\.|,|;|:|!|\?|$)\s*$", text)
        if m:
            pred = m.group(1).upper()

    # 3) Output starts with the option letter (e.g., "B. ...").
    if pred is None:
        m = re.match(r"(?i)^\s*\(?\s*([ABCD])\s*\)?\s*[\.:\-\)]?\s+", text)
        if m:
            pred = m.group(1).upper()

    # 4) Final fallback: last standalone A/B/C/D token anywhere.
    if pred is None:
        matches = list(re.finditer(r"(?i)\b([ABCD])\b", text))
        if matches:
            pred = matches[-1].group(1).upper()

    if pred is None:
        return None, False

    gold = str(mmlu_example["answer"]).strip().upper()
    is_correct = pred == gold
    return pred, is_correct
