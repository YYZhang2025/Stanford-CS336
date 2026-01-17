from pathlib import Path

import regex as re

from cs336_alignment.dataset_utils.math import process_math

DATASET_PROCESSORS = {
    # "gsm8k":
    "math": process_math
}


def convert_cot_to_think_answer(text: str) -> str:
    """
    Convert a chain-of-thought style answer that ends with a line like
    "#### 5" into the desired format by replacing that trailer with
    " </think> <answer> 5 </answer>".

    Examples
    --------
    >>> s = (
    ...     "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\\n"
    ...     "Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\\n"
    ...     "This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\\n"
    ...     "#### 5"
    ... )
    >>> convert_cot_to_think_answer(s)
    "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. </think> <answer> 5 </answer>"

    If no trailing "#### <ans>" is found, this function will try to extract a
    terminal number at the end of the string and use that as the answer. If that
    also fails, the input text is returned unchanged.
    """
    # Match a final line that looks like: #### 5 (possibly with spaces/newline)
    m = re.search(r"####\s*([^\n]+)\s*$", text)
    if m:
        ans = m.group(1).strip()
        prefix = text[: m.start()].rstrip()
        return f"{prefix} </think> <answer>{ans}</answer>"

    # Fallback: try to capture a trailing number at end of text
    m_num = re.search(r"(-?\d+(?:\.\d+)?)\s*$", text)
    if m_num:
        ans = m_num.group(1)
        prefix = text[: m_num.start()].rstrip()
        return f"{prefix} </think> <answer>{ans}</answer>"

    return text


def infer_dataset_name(data_path: str) -> str:
    p = str(Path(data_path)).lower()
    if "math" in p:
        return "math"
    elif "gsm8k" in p:
        return "gsm8k"
    return "math"  # default


def load_and_format_prompts(data_path: str, prompt_template_path: str):
    data_dir = Path(data_path)
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"Data path {data_path} does not exist or is not a directory.")

    # Load prompt template
    with open(prompt_template_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    dataset_name = infer_dataset_name(data_path)
    processor = DATASET_PROCESSORS.get(dataset_name, process_math)
    if dataset_name == "math":
        train_prompts, train_cots, train_answers = processor(data_path, prompt_template=prompt_template)[
            "train"
        ]
        return train_prompts, train_cots, train_answers

    return [], [], []
