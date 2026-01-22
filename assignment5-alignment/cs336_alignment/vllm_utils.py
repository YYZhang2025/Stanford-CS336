from unittest.mock import patch

import torch
from transformers import PreTrainedModel
from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def generate_responses(vllm: LLM, prompts: list[str], sampling_params) -> list[str]:
    outputs = vllm.generate(
        prompts,
        sampling_params=sampling_params,
    )
    responses = [output.outputs[0].text for output in outputs]
    return responses

    # if not sampling_params.logprobs:
    #     responses = [output.outputs[0].text for output in outputs]
    #     return responses
    # else:
    #     assert sampling_params.logprobs == 1, "Only logprobs=1 is supported."
    #     responses = []
    #     gen_ids_list = []
    #     logprobs = []
    #     for output in outputs:
    #         sample_out = output.outputs[0]
    #         responses.append(sample_out.text)
    #         gen_ids = sample_out.token_ids
    #         vllm_logprobs = []
    #         for token_step_logprob_dict in sample_out.logprobs:
    #             for t_id in token_step_logprob_dict:
    #                 vllm_logprobs.append(token_step_logprob_dict[t_id].logprob)

    #         gen_ids_list.append(gen_ids)
    #         logprobs.append(vllm_logprobs)

    #     return responses, gen_ids_list, logprobs
