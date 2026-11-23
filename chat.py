"""Interactive terminal chat with a trained nano GPT checkpoint.

Usage:
    python chat.py
    python chat.py --temperature 0.7 --max-new-tokens 160
"""

import argparse
import os

import torch

from model import GPT
from prompts import STOP_MARKERS, build_prompt, clean_reply
from tokenizer import CharTokenizer

HOLDBACK = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/nano_gpt.pt")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\nRun `python train.py` first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = GPT.from_checkpoint(args.checkpoint, device)
    tokenizer = CharTokenizer(checkpoint["chars"])

    print("Nano GPT chat. Type 'quit' to exit.\n")
    history = []
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in {"quit", "exit"}:
            break
        if not message:
            continue

        prompt = build_prompt(history, message, model.config.block_size, tokenizer).to(device)

        print("Bot: ", end="", flush=True)
        text = ""
        printed = 0
        with torch.no_grad():
            for token in model.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            ):
                text += tokenizer.decode([token.item()])
                if any(marker in text for marker in STOP_MARKERS):
                    break
                safe = len(text) - HOLDBACK
                if safe > printed:
                    print(text[printed:safe], end="", flush=True)
                    printed = safe

        # Account for leading whitespace that got stripped from the reply.
        lead = len(text) - len(text.lstrip())
        reply = clean_reply(text)
        rest_from = max(0, printed - lead)
        if len(reply) > rest_from:
            print(reply[rest_from:], end="")
        print("\n")

        history.append((message, reply))
        history = history[-6:]


if __name__ == "__main__":
    main()
