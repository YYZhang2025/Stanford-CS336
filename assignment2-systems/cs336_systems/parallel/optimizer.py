import torch
from torch.optim import Optimizer
import torch.distributed as dist
from typing import Type, Any



class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.all_params = list(params)
        
        # Create a mapping from parameter ID to its global index
        self.param_to_global_idx = {id(p): i for i, p in enumerate(self.all_params)}
        
        # Get distributed information
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        
        # Each rank is responsible for a shard of parameters
        self.param_shard = self.all_params[self.rank::self.world_size]
        
        # Local optimizer only manages the shard of parameters
        self.local_optimizer = optimizer_cls(self.param_shard, **kwargs)
        
        # Initialize the parent class with the local optimizer's param_groups
        super().__init__(self.local_optimizer.param_groups, self.local_optimizer.defaults)

    @torch.no_grad()
    def step(self, closure=None):
        # 1. Local optimizer performs a step, updating only the local shard of parameters
        loss = self.local_optimizer.step(closure)

        if self.world_size > 1:
            # 2. Synchronize updated parameters across all processes
            for param in self.all_params:
                param_idx = self.param_to_global_idx[id(param)]
                owner_rank = param_idx % self.world_size
                dist.broadcast(param.data, src=owner_rank)
        
        return loss