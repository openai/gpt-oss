import torch
import torch.distributed as dist


def sample_next_token(logits: torch.Tensor, temperature: float) -> int:
    """Sample one token and keep model-parallel ranks on the same token history."""
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0

    if rank == 0:
        if temperature == 0.0:
            token = torch.argmax(logits, dim=-1).to(torch.long)
        else:
            probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
            token = torch.multinomial(probs, num_samples=1).squeeze(0)
    else:
        token = torch.empty((), dtype=torch.long, device=logits.device)

    if distributed:
        dist.broadcast(token, src=0)

    return int(token.item())
