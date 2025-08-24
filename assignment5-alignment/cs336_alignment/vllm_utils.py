from unittest.mock import patch

import torch
from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.utils import print_color


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


def load_model_into_vllm_instance(model: torch.nn.Module, llm: LLM):
    # snapshot to CPU -> then load into vLLM
    model.eval()
    model.tie_weights()
    cpu_sd = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(cpu_sd.items())
    model.train()
    torch.cuda.synchronize(torch.device("cuda:1"))
    print_color("Model weights loaded into VLLM instance.")
