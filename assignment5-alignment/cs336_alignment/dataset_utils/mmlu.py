import json
import os

import pandas as pd

MMLU_TEMPLATE = """ {question}\nA. {option_A}\nB. {option_B}\nC. {option_C}\nD. {option_D}"""


def collect_rows(data_dir: str) -> list[dict]:
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    question_answers = []
    for filename in csv_files:
        p = os.path.join(data_dir, filename)
        df = pd.read_csv(p, header=None)
        for _, row in df.iterrows():
            question = MMLU_TEMPLATE.format(
                question=row[0],
                option_A=row[1],
                option_B=row[2],
                option_C=row[3],
                option_D=row[4],
            )
            answer = row[5].strip()
            question_answers.append({"question": question, "answer": answer})
    return question_answers


def process_row(row: dict) -> tuple[str, str]:
    question = row["question"]
    answer = row["answer"]

    return question, str(answer).lower()
