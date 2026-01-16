from __future__ import annotations

from typing import Any, Type

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """Optimizer wrapper that shards optimizer *state* across ranks.

    Each rank owns (and thus holds optimizer state for) a subset of parameters.
    After `step()`, updated parameters are broadcast from their owner ranks so that
    all ranks keep identical model weights.

    Public interface required by the assignment:
      - __init__(params, optimizer_cls, **kwargs)
      - step(closure=None, **kwargs)
      - add_param_group(param_group)

    Sharding rule:
      owner_rank = global_param_index % world_size
      where global_param_index is assigned in the order params/groups are added.
    """

    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        # Distributed info
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict(kwargs)

        # Global (replicated) parameter list + ownership mapping
        self.all_params: list[torch.nn.Parameter] = []
        self.param_to_global_idx: dict[int, int] = {}

        # Local wrapped optimizer (only created if/when this rank has params)
        self.local_optimizer: Optimizer | None = None

        # IMPORTANT: call Optimizer constructor.
        # This will call our overridden add_param_group() for each group in `params`.
        super().__init__(params, defaults=self.optimizer_kwargs)

        # If this rank never owned any params, local_optimizer stays None.
        # That's okay: step() becomes a no-op locally, but we still participate in broadcasts.

    def _owner_rank(self, param: torch.nn.Parameter) -> int:
        idx = self.param_to_global_idx[id(param)]
        return idx % self.world_size

    def _assign_and_track_params(self, group_params: list[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
        """Track new params in global order and return the subset owned by this rank."""
        local: list[torch.nn.Parameter] = []
        for p in group_params:
            pid = id(p)
            if pid not in self.param_to_global_idx:
                gidx = len(self.all_params)
                self.all_params.append(p)
                self.param_to_global_idx[pid] = gidx

            # Determine ownership based on global index
            if self.world_size == 1 or self._owner_rank(p) == self.rank:
                local.append(p)
        return local

    def add_param_group(self, param_group: dict[str, Any]):
        if "params" not in param_group:
            raise ValueError("param_group must have a 'params' key")

        # Normalize params to a list[Parameter]
        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            raise TypeError("param_group['params'] must be an iterable of Parameters, not a Tensor")
        group_params = list(params)  # materialize

        # Track global params and select this-rank shard
        local_params = self._assign_and_track_params(group_params)

        # Build the local param_group dict (copy hyperparams, but shard params)
        local_group = {k: v for k, v in param_group.items() if k != "params"}
        local_group["params"] = local_params

        # If this rank owns no params in this group, we skip adding it locally.
        # Other ranks may still own params here.
        if len(local_params) == 0:
            return

        # 1) Register the local group on THIS Optimizer wrapper
        super().add_param_group(local_group)

        # 2) Ensure the wrapped local optimizer exists and has the same group
        if self.local_optimizer is None:
            # Create the local optimizer using current local param_groups
            self.local_optimizer = self.optimizer_cls(self.param_groups, **self.optimizer_kwargs)
        else:
            # Keep local optimizer groups in sync
            self.local_optimizer.add_param_group(local_group)

    @torch.no_grad()
    def step(self, closure=None, **kwargs: Any):
        """Run a sharded optimizer step then synchronize updated params across ranks."""
        loss = None
        if self.local_optimizer is not None:
            loss = self.local_optimizer.step(closure=closure, **kwargs)

        if self.world_size > 1:
            # Synchronize updated parameters across all ranks.
            # Each parameter is broadcast from its owner rank.
            for p in self.all_params:
                dist.broadcast(p.data, src=self._owner_rank(p))

        return loss
