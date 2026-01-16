import torch
import torch.distributed as dist
import torch.nn as nn

from .utils import broadcast_model_from_rank0


class DDPBucket(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float = 25.0):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Distributed package is not available or initialized.")
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.module = module
        broadcast_model_from_rank0(self.module)

        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        if self.bucket_size_bytes <= 0:
            raise ValueError("bucket_size_mb must be > 0")

        # Buckets are lists of parameters, plus a flat buffer and bookkeeping.
        self.buckets: list[list[nn.Parameter]] = []
        self.bucket_bufs: list[torch.Tensor] = []
        self.bucket_expected: list[int] = []
        self.bucket_ready: list[int] = []
        self.bucket_work: list[dist.Work | None] = []

        # Map param -> (bucket_id, offset)
        self._meta: dict[int, tuple[int, int]] = {}
        self._create_buckets()
        self._register_hooks()

    def _create_buckets(self) -> None:
        # Reverse order improves overlap because grads become ready roughly in that order.
        params = [p for p in self.module.parameters() if p.requires_grad]
        params = list(reversed(params))

        cur: list[nn.Parameter] = []
        cur_bytes = 0

        def flush():
            if not cur:
                return
            bucket_id = len(self.buckets)
            self.buckets.append(cur.copy())

            # Allocate a flat buffer for this bucket
            total = sum(p.numel() for p in cur)
            device = cur[0].device
            dtype = cur[0].dtype
            buf = torch.empty(total, device=device, dtype=dtype)
            self.bucket_bufs.append(buf)

            # Assign offsets into the flat buffer
            off = 0
            for p in cur:
                self._meta[id(p)] = (bucket_id, off)
                off += p.numel()

            self.bucket_expected.append(len(cur))
            self.bucket_ready.append(0)
            self.bucket_work.append(None)

            cur.clear()

        for p in params:
            p_bytes = p.numel() * p.element_size()
            if cur and cur_bytes + p_bytes > self.bucket_size_bytes:
                flush()
                cur_bytes = 0

            cur.append(p)
            cur_bytes += p_bytes

            # If one param is already large, keep it alone.
            if cur_bytes >= self.bucket_size_bytes:
                flush()
                cur_bytes = 0

        flush()

    def _register_hooks(self) -> None:
        # Hook returns a view into the bucket buffer so param.grad becomes that view.
        for p in self.module.parameters():
            if not p.requires_grad:
                continue
            if id(p) not in self._meta:
                continue

            bucket_id, offset = self._meta[id(p)]
            numel = p.numel()
            shape = p.shape

            def make_hook(bid: int, off: int, n: int, shp: torch.Size):
                def hook(grad: torch.Tensor):
                    buf = self.bucket_bufs[bid]
                    view = buf.narrow(0, off, n).view(shp)
                    view.copy_(grad)

                    self.bucket_ready[bid] += 1
                    if self.bucket_ready[bid] == self.bucket_expected[bid]:
                        # Launch async all-reduce on the full bucket
                        self.bucket_work[bid] = dist.all_reduce(
                            self.bucket_bufs[bid],
                            op=dist.ReduceOp.SUM,
                            async_op=True,
                        )
                    return view

                return hook

            p.register_hook(make_hook(bucket_id, offset, numel, shape))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        if self.world_size == 1:
            # reset
            for i in range(len(self.bucket_ready)):
                self.bucket_ready[i] = 0
                self.bucket_work[i] = None
            return
        # Wait & average per bucket
        for i, work in enumerate(self.bucket_work):
            if work is not None:
                work.wait()
                self.bucket_bufs[i].div_(self.world_size)

                # reset for next iteration
            self.bucket_ready[i] = 0
            self.bucket_work[i] = None
