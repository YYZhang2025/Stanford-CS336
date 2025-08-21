import json

import regex as re


def wrap_prompt(text: str, prompt_path: str):
    with open(prompt_path, "r") as file:
        prompt = file.read()
    return prompt.format(question=text)


def extract_reference_answer(answer: str) -> str:
    ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"


def load_and_format_prompts(data_path: str, prompt_path: str):
    ANSWER_WRAP = "</think> <answer>{answer}</answer>"
    with open(prompt_path, "r") as file:
        prompt = file.read()
    prompts = []
    answers = []
    with open(data_path, "r") as file:
        for line in file:
            data = json.loads(line)
            prompts.append(prompt.format(question=data["question"]))
            answers.append(ANSWER_WRAP.format(answer=data["answer"]))

    return prompts, answers


def load_json_to_list(file_path: str) -> list[dict]:
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line.strip()))

    return data_list
