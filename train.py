"""Train the nano GPT model on the conversation corpus in data/.

Usage:
    python train.py                      # defaults, CPU friendly
    python train.py --max-steps 5000
"""

import argparse
import glob
import math
import os
import time

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def load_corpus(data_dir="data"):
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not files:
        raise SystemExit(f"No .txt files found in {data_dir}/")
    parts = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            parts.append(f.read().strip() + "\n")
    return "\n".join(parts)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, iters=20):
    model.eval()
    losses = torch.zeros(iters)
    for i in range(iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="checkpoints/nano_gpt.pt")
    parser.add_argument("--block-size", type=int, default=160)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=1400)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    text = load_corpus(args.data_dir)
    tokenizer = CharTokenizer.from_text(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(0.95 * len(ids))
    train_data, val_data = ids[:split], ids[split:]

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )

    print(f"device: {device}")
    print(f"corpus: {len(text):,} characters, vocab: {tokenizer.vocab_size}")
    print(f"model: {model.num_params():,} parameters")

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))

    def save_checkpoint(step, val_loss):
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": config.__dict__,
                "chars": tokenizer.chars,
                "step": step,
                "val_loss": val_loss,
            },
            args.out,
        )
        tokenizer.save(os.path.join(os.path.dirname(args.out), "tokenizer.json"))

    start = time.time()
    losses = []
    best_val = float("inf")
    for step in range(args.max_steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)

        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0 or step == args.max_steps - 1:
            losses.append(loss.item())

        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            train_loss = estimate_loss(
                model, train_data, args.block_size, args.batch_size, device, args.eval_iters
            )
            val_loss = estimate_loss(
                model, val_data, args.block_size, args.batch_size, device, args.eval_iters
            )
            elapsed = time.time() - start
            marker = ""
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(step, val_loss)
                marker = f"  <- best, saved {args.out}"
            print(
                f"step {step:5d} | train {train_loss:.4f} | val {val_loss:.4f} "
                f"| lr {lr_at(step):.2e} | {elapsed:6.1f}s{marker}"
            )

    print(f"best val loss {best_val:.4f}; checkpoint saved to {args.out}")


if __name__ == "__main__":
    main()
