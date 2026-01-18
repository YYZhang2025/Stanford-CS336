import json

import torch
from torch.utils.data import Dataset


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict:
    prompt_tokens = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )

    output_tokens = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )

    input_ids = []
    response_mask = []

    for p_ids, o_ids in zip(prompt_tokens["input_ids"], output_tokens["input_ids"]):
        combined_ids = p_ids + o_ids
        input_ids.append(combined_ids)

        mask = ([False] * len(p_ids)) + ([True] * len(o_ids))
        response_mask.append(mask)

    max_len = max(len(ids) for ids in input_ids)
    pad_id = tokenizer.pad_token_id

    def pad_to(x, value):
        return x + [value] * (max_len - len(x))

    full = torch.tensor([pad_to(x, pad_id) for x in input_ids], dtype=torch.long)
    input_ids = full[:, :-1].contiguous()
    labels = full[:, 1:].contiguous()
    response_mask = torch.tensor([pad_to(x, False) for x in response_mask], dtype=torch.bool)[
        :, 1:
    ].contiguous()

    assert input_ids.shape == labels.shape == response_mask.shape, (
        "Shapes of input_ids, labels, and response_mask must match"
    )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


class ReasoningDataset(Dataset):
    def __init__(self, questions: list[str], cots: list[str], answers: list[str], prompt_template_path: str):
        self.questions = questions
        self.cots = cots
        self.answers = answers

        with open(prompt_template_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        self.prompts = [self.prompt_template.format(question=q) for q in self.questions]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        cot = self.cots[idx]
        answer = self.answers[idx]

        return prompt, cot, answer

    @classmethod
    def load_from_disk(cls, path: str, prompt_template_path: str):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        questions = []
        cots = []
        answers = []
        for row in rows:
            questions.append(row["question"])
            cots.append(row["cot"])
            answers.append(row["answer"])

        return cls(questions, cots, answers, prompt_template_path=prompt_template_path)


def collate_fn(batch, tokenizer):
    """
    return:
        {
            "input_ids": input_ids,
            "labels": labels,
            "response_mask": response_mask,
        }
    """
    prompts, cots, answers = zip(*batch)
    tokenized = tokenize_prompt_and_output(
        prompt_strs=list(prompts),
        output_strs=list(cots),
        tokenizer=tokenizer,
    )

    tokenized["prompts"] = list(prompts)
    tokenized["answers"] = list(answers)

    return tokenized
