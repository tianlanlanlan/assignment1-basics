import argparse
import re

import torch

from cs336_basics.bpe import Tokenizer
from cs336_basics.generation import generate
from cs336_basics.transformer_lm import TransformerLM


def infer_model_config(state_dict: dict[str, torch.Tensor]) -> dict:
    """Infer model hyperparameters from a checkpoint state dict."""
    config = {}

    # vocab_size and d_model from token_embedding weights
    emb_weight = state_dict["token_embedding.weights"]
    config["vocab_size"] = emb_weight.shape[0]
    config["d_model"] = emb_weight.shape[1]

    # num_layers from matching layer keys
    layer_ids = set()
    for key in state_dict:
        m = re.match(r"layers\.(\d+)\.", key)
        if m:
            layer_ids.add(int(m.group(1)))
    config["num_layers"] = max(layer_ids) + 1 if layer_ids else 0

    # d_ff from the first FFN layer's w1 weights
    w1_key = next(k for k in state_dict if k.startswith("layers.0.ffn.w1"))
    config["d_ff"] = state_dict[w1_key].shape[0]

    # context_length and num_heads from RoPE cos buffer
    cos_key = next(k for k in state_dict if k.endswith("rope.cos"))
    cos = state_dict[cos_key]
    config["context_length"] = cos.shape[0]
    d_k = cos.shape[-1] * 2
    config["num_heads"] = config["d_model"] // d_k

    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer language model.")

    # Checkpoint (only required argument)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to model checkpoint (.pt)")

    # Optional overrides for model config (auto-detected from checkpoint)
    parser.add_argument("--vocab-size", type=int, default=None, help="Override auto-detected vocab size")
    parser.add_argument("--context-length", type=int, default=None, help="Override auto-detected context length")
    parser.add_argument("--d-model", type=int, default=None, help="Override auto-detected d_model")
    parser.add_argument("--num-layers", type=int, default=None, help="Override auto-detected num_layers")
    parser.add_argument("--num-heads", type=int, default=None, help="Override auto-detected num_heads")
    parser.add_argument("--d-ff", type=int, default=None, help="Override auto-detected d_ff")
    parser.add_argument(
        "--rope-theta", type=float, default=10000.0, help="RoPE theta (not in checkpoint, defaults to 10000.0)"
    )

    # Tokenizer
    parser.add_argument(
        "--vocab-json", type=str, default="data/TinyStoriesV2-GPT4-train.vocab.json", help="Path to vocab.json"
    )
    parser.add_argument(
        "--merges-txt", type=str, default="data/TinyStoriesV2-GPT4-train.merges.txt", help="Path to merges.txt"
    )

    # Generation parameters
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt text")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p (nucleus) sampling threshold")

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)

    # Load checkpoint first to infer model config
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    inferred = infer_model_config(state_dict)

    # Use inferred values unless overridden
    vocab_size = args.vocab_size or inferred["vocab_size"]
    context_length = args.context_length or inferred["context_length"]
    d_model = args.d_model or inferred["d_model"]
    num_layers = args.num_layers or inferred["num_layers"]
    num_heads = args.num_heads or inferred["num_heads"]
    d_ff = args.d_ff or inferred["d_ff"]
    rope_theta = args.rope_theta

    print(
        f"Model config: vocab_size={vocab_size}, d_model={d_model}, num_layers={num_layers}, "
        f"num_heads={num_heads}, d_ff={d_ff}, context_length={context_length}"
    )

    # Load tokenizer
    tokenizer = Tokenizer.from_files(args.vocab_json, args.merges_txt, special_tokens=["<|endoftext|>"])

    # Initialize model and load weights
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    model.load_state_dict(state_dict)
    iteration = checkpoint.get("iteration", 0)
    print(f"Loaded checkpoint from iteration {iteration}")

    # Generate
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        context_length=context_length,
        device=device,
    )

    print(f"\n--- Generated text ---\n{output}\n")


if __name__ == "__main__":
    main()
