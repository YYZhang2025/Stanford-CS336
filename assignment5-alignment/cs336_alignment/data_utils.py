import json

import regex as re


# --- Chain-of-thought to <think> <answer> conversion utility ---
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


def extract_reference_answer(response: str) -> str:
    from cs336_alignment.drgrpo_grader import extract_answer

    model_answer = response.split("<answer>")[-1].replace("</answer>", "")
    if "\\boxed" in model_answer:
        model_answer = extract_answer(model_answer)

    return model_answer


def extract_gsm8k_answer(answer: str) -> str:
    ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"


def wrap_prompt(text: str, prompt_path: str):
    with open(prompt_path, "r") as file:
        prompt = file.read()
    return prompt.format(question=text)


def load_and_format_prompts(data_path: str, prompt_path: str) -> tuple[list[str], list[str], list[str]]:
    with open(prompt_path, "r") as file:
        prompt = file.read()

    prompts = []
    cot = []
    answers = []
    with open(data_path, "r") as file:
        for line in file:
            data = json.loads(line)
            prompts.append(prompt.format(question=data["question"]))
            cot.append(convert_cot_to_think_answer(data["answer"]))
            answers.append(extract_gsm8k_answer(data["answer"]))

    return prompts, cot, answers


def load_json_to_list(file_path: str) -> list[dict]:
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line.strip()))

    return data_list
