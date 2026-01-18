from cs336_alignment.drgrpo_grader import extract_answer, r1_zero_reward_fn
from cs336_alignment.vllm_utils import generate_responses


def extract_reference_answer(response: str) -> str:
    model_answer = response.split("<answer>")[-1].replace("</answer>", "")
    if "\\boxed" in model_answer:
        model_answer = extract_answer(model_answer)

    return model_answer


def evaluate_responses(
    vllm: LLM,
    prompts: list[str],
    answers: list[str],
    sampling_params,
):
    responses = generate_responses(vllm, prompts, sampling_params)
    allinfo_list = []
    for response, answer, prompt in zip(responses, answers, prompts):
        reward_dict = r1_zero_reward_fn(response, ground_truth=answer)
        allinfo_list.append(reward_dict)

    # Gather statistics
    overview = {
        "total": len(responses),
        "correct": 0,
        "format_wrong": 0,
        "answer_wrong": 0,
    }

    for reward_dict in allinfo_list:
        if reward_dict["reward"] == 1:
            overview["correct"] += 1
        elif not reward_dict["format_reward"]:
            overview["format_wrong"] += 1
        else:
            overview["answer_wrong"] += 1

    return overview
