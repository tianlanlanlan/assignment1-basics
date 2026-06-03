import torch

from cs336_basics.softmax import softmax


def temperature_softmax(
    logits: torch.Tensor, temperature: float, dim: int = -1
) -> torch.Tensor:
    if temperature == 0:
        return logits
    return softmax(logits / temperature, dim=dim)


def top_p_filtering(probs: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Keep only tokens with cumulative prob < p, but always keep at least one
    mask = cumulative_probs > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_probs[mask] = 0.0

    # Re-normalize
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # Scatter back to original index order
    return torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)


def sample_next_token(
    logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0
) -> torch.Tensor:
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    probs = temperature_softmax(logits, temperature)
    probs = top_p_filtering(probs, top_p)

    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    context_length: int = 128,
    device: str = "cpu",
) -> str:
    model.eval()
    model.to(device)

    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    if eos_token_id is None:
        eos_token_id = tokenizer.special_token_to_id.get("<|endoftext|>")

    # Ensure we don't exceed context_length
    if len(input_ids) > context_length:
        input_ids = input_ids[-context_length:]

    for _ in range(max_length):
        # Truncate to context_length if needed
        if len(input_ids) > context_length:
            input_ids = input_ids[-context_length:]

        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        logits = model(x)  # (1, seq_len, vocab_size)
        next_logits = logits[0, -1, :]  # (vocab_size,)

        next_id = sample_next_token(next_logits, temperature, top_p)
        next_id = next_id.item()

        input_ids.append(next_id)

        if next_id == eos_token_id:
            break

    return tokenizer.decode(input_ids)
