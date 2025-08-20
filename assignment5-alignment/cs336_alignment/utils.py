import json

import regex as re

ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")


def extract_reference_answer(answer: str, pattern: re.Pattern = ANS_RE) -> str:
    match = pattern.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"


def load_json_to_list(file_path: str) -> list[str]:
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line.strip()))

    return data_list
