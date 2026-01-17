# from unittest.mock import patch

# import torch
# from transformers import PreTrainedModel
# from vllm import LLM
# from vllm.model_executor import set_random_seed as vllm_set_random_seed


# def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
#     vllm_set_random_seed(seed)
#     world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
#     profiling_patch = patch(
#         "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
#     )
#     with world_size_patch, profiling_patch:
#         return LLM(
#             model=model_id,
#             device=device,
#             dtype=torch.bfloat16,
#             enable_prefix_caching=True,
#             gpu_memory_utilization=gpu_memory_utilization,
#         )


# def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
#     """
#     Copied from https://github.com/huggingface/trl/blob/
#     22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
#     """
#     state_dict = policy.state_dict()
#     llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
#     llm_model.load_weights(state_dict.items())
