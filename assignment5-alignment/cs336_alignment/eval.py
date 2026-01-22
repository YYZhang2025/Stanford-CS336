from cs336_alignment.drgrpo_grader import extract_answer, r1_zero_reward_fn
from cs336_alignment.vllm_utils import generate_responses


def extract_reference_answer(response: str) -> str:
    model_answer = response.split("<answer>")[-1].replace("</answer>", "")
    if "\\boxed" in model_answer:
        model_answer = extract_answer(model_answer)

    return model_answer


def evaluate_responses(vllm, prompts, answers, sampling_params):
    responses = generate_responses(vllm, prompts, sampling_params)

    # Safety: avoid silent truncation if lengths mismatch
    assert len(responses) == len(answers) == len(prompts)

    overview = {
        "total": len(responses),
        "answer_correct": 0,
        "format_correct": 0,
        "reward_1": 0,
        "formatted_but_answer_wrong": 0,
        "answer_accuracy": 0.0,
    }

    for response, gt in zip(responses, answers):
        r = r1_zero_reward_fn(response, ground_truth=gt)

        if r["format_reward"] == 1.0:
            overview["format_correct"] += 1

        if r["answer_reward"] == 1.0:
            overview["answer_correct"] += 1
        elif r["format_reward"] == 1.0:
            overview["formatted_but_answer_wrong"] += 1

        if r["reward"] == 1.0:
            overview["reward_1"] += 1

    overview["answer_accuracy"] = overview["answer_correct"] / overview["total"]
    return overview
