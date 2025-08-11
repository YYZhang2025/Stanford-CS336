import torch 
import torch.nn as nn 
import torch.distributed as dist 

class DDPModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        
        
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

            for param in self.model.parameters():
                dist.broadcast(param.data, src=0)
        else:
            self.rank = 0
            self.world_size = 1
            
            
        self.handles = []
        for p in self.model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, p):
        if p.grad is None or self.world_size <= 1:
            return

        handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append(handle)
        
    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)
    
    
    def finish_gradient_synchronization(self):
        if self.world_size <= 1:
            return

        for handle in self.handles:
            handle.wait()
            
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad /= self.world_size
        
        self.handles.clear()
        


class DDPBucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        
        super().__init__()
        
        self.module = module
        
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        
        for p in self.module.parameters():
            if dist.is_initialized():
                dist.broadcast(p.data, src=0)   

        
        self.buckets = []
        self.param_to_bucket = {}
        
        
        visited = set()
        
        params_in_reverse = list(self.module.parameters())[::-1]
        
        curr_bucket = []
        curr_size = 0
        bucket_size_mb = bucket_size_mb * 1024 * 1024  # Convert to bytes
        
        
        for p in params_in_reverse:
            if p.requires_grad and id(p) not in visited:
                visited.add(id(p))
                
                if p.grad is None:
                    continue
                
                param_size = p.numel() * p.element_size()
                
                if curr_size + param_size > bucket_size_mb:
                    if curr_bucket:
                        self.buckets.append(curr_bucket)
                    curr_bucket = [p]
                    curr_size = param_size
                else:
                    curr_bucket.append(p)
                    curr_size += param_size
                
        
        if curr_bucket:
            self.buckets.append(curr_bucket)
            
        
        for i, bucket in enumerate(self.buckets):
            for param in bucket:
                self.param_to_bucket[id(param)] = i
        
        self.bucket_grad_ready_counts = [0] * len(self.buckets)
        self.handles = []

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._grad_hook)
        
    def _grad_hook(self, param):
        if param.grad is None or self.world_size <= 1:
            return

        # 使用参数ID来查找桶
        bucket_index = self.param_to_bucket[id(param)]
        bucket = self.buckets[bucket_index]
        
        self.bucket_grad_ready_counts[bucket_index] += 1

        if self.bucket_grad_ready_counts[bucket_index] == len(bucket):
            grads_to_reduce = [p.grad for p in bucket]
            flat_grads = torch._utils._flatten_dense_tensors(grads_to_reduce)
            
            handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append((handle, flat_grads, bucket))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        if self.world_size <= 1:
            return

        for handle, flat_grads, bucket in self.handles:
            handle.wait()
            flat_grads /= self.world_size
            
            unflattened_grads = torch._utils._unflatten_dense_tensors(flat_grads, bucket)
            for param, grad in zip(bucket, unflattened_grads):
                param.grad = grad

        self.handles.clear()
        self.bucket_grad_ready_counts = [0] * len(self.buckets)