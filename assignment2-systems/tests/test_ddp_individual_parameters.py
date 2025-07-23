import logging
from copy import deepcopy
from typing import Type

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from .adapters import (
    ddp_individual_parameters_on_after_backward,
    get_ddp_individual_parameters,
)
from .common import (
    FIXTURES_PATH,
    ToyModel,
    ToyModelWithTiedWeights,
    _cleanup_process_group,
    _setup_process_group,
    validate_ddp_net_equivalence,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("model_class", [ToyModel, ToyModelWithTiedWeights])
def test_DistributedDataParallelIndividualParameters(model_class):
    world_size = 2
    mp.spawn(
        _test_DistributedDataParallelIndividualParameters,
        args=(world_size, model_class),
        nprocs=world_size,
        join=True,
    )


def _test_DistributedDataParallelIndividualParameters(rank: int, world_size: int, model_class: Type[torch.nn.Module]):
    # Use gloo backend for CPU
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    # Execute barrier prior to running test to ensure that every process
    # has finished initialization and that the following test
    # immediately exiting due to a skip doesn't cause flakiness.
    dist.barrier()

    # Seed to ensure that ranks are initialized with different initial models.
    torch.manual_seed(rank)

    # Create a toy model and move it to the proper device.
    # This is our non-parallel baseline.
    non_parallel_model = model_class().to(device)

    # Create a DDP model. Note that the weights of this model should
    # match the non-parallel baseline above.
    ddp_base = deepcopy(non_parallel_model)
    ddp_model = get_ddp_individual_parameters(ddp_base)

    # If we're on rank 0, the DDP model should still exactly match the parameters of the
    # non-parallel baseline (since the parameters on rank 0 weren't changed).
    # If we're not on rank 0, the DDP model's parameters should have been updated with
    # the parameters from rank 0. So, double-check that the parameter differ from the
    # local initial state.
    for (non_parallel_param_name, non_parallel_model_parameter), (
        ddp_model_param_name,
        ddp_model_parameter,
    ) in zip(non_parallel_model.named_parameters(), ddp_model.named_parameters()):
        # This parameter was initialized as [2, 2], so we expect its value to remain the same
        is_no_grad_fixed_param = (
            "no_grad_fixed_param" in ddp_model_param_name or "no_grad_fixed_param" in non_parallel_param_name
        )
        if rank == 0 or is_no_grad_fixed_param:
            assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
        else:
            assert not torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

    # Make sure all the ranks have the same model state
    validate_ddp_net_equivalence(ddp_model)

    # Load the dataset from disk, so we can ensure that every rank has the same
    # overall pool of data.
    # Shape: (20, 10)
    all_x = torch.load(FIXTURES_PATH / "ddp_test_data.pt")
    # Shape: (20, 5)
    all_y = torch.load(FIXTURES_PATH / "ddp_test_labels.pt")

    # Each rank will see only 10 examples (out of the total dataset size of 20)
    assert all_x.size(0) % world_size == 0
    local_bs = int(all_y.size(0) / world_size)

    loss_fn = nn.MSELoss()

    # Optimizer for the DDP model
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)
    # Optimizer for the non-parallel model
    non_parallel_optimizer = optim.SGD(non_parallel_model.parameters(), lr=0.1)

    for i in range(5):
        ddp_optimizer.zero_grad()
        non_parallel_optimizer.zero_grad()

        # Run the non-parallel model on all the data and take a gradient step
        non_parallel_data = all_x.to(device)
        non_parallel_labels = all_y.to(device)
        non_parallel_outputs = non_parallel_model(non_parallel_data)
        non_parallel_loss = loss_fn(non_parallel_outputs, non_parallel_labels)
        non_parallel_loss.backward()
        non_parallel_optimizer.step()

        # At this point, the parameters of non-parallel model should differ
        # from the parameters of the DDP model (since we've applied the
        # gradient step to the non-parallel model, but not to the DDP model).
        if rank == 0:
            for non_parallel_model_parameter, ddp_model_parameter in zip(
                non_parallel_model.parameters(), ddp_model.parameters()
            ):
                if non_parallel_model_parameter.requires_grad and ddp_model_parameter.requires_grad:
                    # The only parameters that change are those that require_grad
                    assert not torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
                else:
                    # parameters that don't require_grad shouldn't change
                    assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

        # While the non-parallel model does a forward pass on all the data (20 examples),
        # each DDP rank only sees 10 (disjoint) examples.
        # However, the end result should be the same as doing a forward pass on all 20 examples.
        offset = rank * local_bs
        ddp_data = all_x[offset : offset + local_bs, :].to(device)
        ddp_labels = all_y[offset : offset + local_bs, :].to(device)
        ddp_outputs = ddp_model(ddp_data)
        ddp_loss = loss_fn(ddp_outputs, ddp_labels)
        ddp_loss.backward()

        # Run student-written code that needs to execute after the backward pass,
        # but before the optimizer step (e.g., to wait for all DDP ranks to sync gradients)
        ddp_individual_parameters_on_after_backward(ddp_model, ddp_optimizer)

        ddp_optimizer.step()

        # At this point, the non-parallel model should exactly match the parameters of the DDP model
        if rank == 0:
            for non_parallel_model_parameter, ddp_model_parameter in zip(
                non_parallel_model.parameters(), ddp_model.parameters()
            ):
                assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

        # Shuffle the data so that during the next iteration, each DDP rank sees a different set of inputs.
        # We make sure to use the same seed when shuffling (else the per-rank examples might not be disjoint).
        torch.manual_seed(42 + i)
        shuffle_idxs = torch.randperm(all_x.size(0))
        all_x = all_x[shuffle_idxs]
        all_y = all_y[shuffle_idxs]

    # After training is done, we should have the same weights on both the non-parallel baseline
    # and the model trained with DDP.
    if rank == 0:
        for non_parallel_model_parameter, ddp_model_parameter in zip(
            non_parallel_model.parameters(), ddp_model.parameters()
        ):
            assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
    _cleanup_process_group()
