import os
import torch
import torch.distributed as dist


def suppress_output(rank):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force:
            builtin_print("rank #%d:" % rank, *args, **kwargs)
        elif rank == 0:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed() -> torch.device:
    """Initialize the model for distributed inference."""
    # Initialize distributed inference
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=rank
        )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Warm up NCCL to avoid first-time latency
    if world_size > 1:
        x = torch.ones(1, device=device)
        dist.all_reduce(x)
        torch.cuda.synchronize(device)

    suppress_output(rank)
    return device
