def wrap_cot_with_answer(cot: str, answer: str) -> str:
    return f"{cot}\n</think> <answer>{str(answer)}</answer>"
