import argparse
import math
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import get_batch
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics.transformer_lm import TransformerLM


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")

    # Model hyperparameters
    parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size")
    parser.add_argument("--context-length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--d-model", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=512, help="Feedforward hidden dimension")
    parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta parameter")

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-iters", type=int, default=50000, help="Total number of training iterations")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min-learning-rate", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--warmup-iters", type=int, default=2000, help="Warmup iterations")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient L2 norm")

    # Data
    parser.add_argument(
        "--train-data", type=str, required=True, help="Path to training data (flat binary of token IDs)"
    )
    parser.add_argument("--val-data", type=str, default=None, help="Path to validation data (flat binary of token IDs)")
    parser.add_argument("--dtype", type=str, default="uint16", help="Data type of the token ID binary files")

    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save-interval", type=int, default=5000, help="Iterations between checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Logging
    parser.add_argument("--log-interval", type=int, default=10, help="Iterations between logging")
    parser.add_argument("--val-interval", type=int, default=500, help="Iterations between validation")
    parser.add_argument("--val-iters", type=int, default=200, help="Number of validation batches to evaluate")
    parser.add_argument("--wandb-project", type=str, default=None, help="wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity name")

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")

    return parser.parse_args()


@torch.no_grad()
def evaluate(model, val_data, batch_size, context_length, val_iters, device):
    model.eval()
    total_loss = 0.0
    for _ in range(val_iters):
        xs, ys = get_batch(val_data, batch_size, context_length, device)
        logits = model(xs)
        loss = cross_entropy(logits.view(-1, logits.shape[-1]), ys.view(-1))
        total_loss += loss.item()
    avg_loss = total_loss / val_iters
    perplexity = math.exp(avg_loss)
    model.train()
    return avg_loss, perplexity


def main():
    args = parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device)

    # Load data with memmap
    train_data = np.memmap(args.train_data, dtype=args.dtype, mode="r")
    val_data = None
    if args.val_data is not None:
        val_data = np.memmap(args.val_data, dtype=args.dtype, mode="r")

    # Initialize model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # Initialize wandb if requested
    if args.wandb_project is not None:
        import wandb

        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

    # Resume from checkpoint if specified
    iteration = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        iteration = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from checkpoint at iteration {iteration}")
        iteration += 1

    # Training loop
    model.train()
    total_tokens = 0
    total_time = 0.0
    pbar = tqdm(total=args.max_iters, initial=iteration, desc="Training", dynamic_ncols=True)

    while iteration < args.max_iters:
        iter_start = time.time()

        # Get learning rate for this iteration
        lr = get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get batch
        xs, ys = get_batch(train_data, args.batch_size, args.context_length, device)

        # Forward pass
        logits = model(xs)
        loss = cross_entropy(logits.view(-1, logits.shape[-1]), ys.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        gradient_clipping(model.parameters(), args.grad_clip)

        # Optimizer step
        optimizer.step()

        iter_time = time.time() - iter_start
        total_time += iter_time
        total_tokens += args.batch_size * args.context_length

        # Logging
        if iteration % args.log_interval == 0:
            metrics = {
                "loss": loss.item(),
                "perplexity": math.exp(loss.item()),
                "lr": lr,
                "iter": iteration,
                "tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
                "ms_per_iter": total_time * 1000 / max(1, iteration - (0 if args.resume is None else 0)),
            }
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "ppl": f"{math.exp(loss.item()):.2f}"})
            if args.wandb_project is not None:
                wandb.log(metrics, step=iteration)

        # Validation
        if val_data is not None and iteration % args.val_interval == 0 and iteration > 0:
            val_loss, val_ppl = evaluate(model, val_data, args.batch_size, args.context_length, args.val_iters, device)
            val_metrics = {"val_loss": val_loss, "val_perplexity": val_ppl}
            if args.wandb_project is not None:
                wandb.log(val_metrics, step=iteration)
            tqdm.write(f"Step {iteration}: val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.save_dir, "best.pt")
                save_checkpoint(model, optimizer, iteration, best_path)

        # Save checkpoint
        if iteration % args.save_interval == 0 and iteration > 0:
            ckpt_path = os.path.join(args.save_dir, f"iter_{iteration}.pt")
            save_checkpoint(model, optimizer, iteration, ckpt_path)

        iteration += 1
        pbar.update(1)

    pbar.close()

    # Save final checkpoint
    final_path = os.path.join(args.save_dir, "final.pt")
    save_checkpoint(model, optimizer, iteration, final_path)
    print(f"Training complete. Final checkpoint saved to {final_path}")

    if args.wandb_project is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
