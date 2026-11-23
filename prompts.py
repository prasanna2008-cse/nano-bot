"""Prompt formatting shared by the CLI and the web server."""

import torch

STOP_MARKERS = ("\nUser:", "\nBot:", "\n\n")


def build_prompt(history, message, block_size, tokenizer):
    """Turn chat history + new message into a [1, T] tensor of token ids.

    The training corpus uses this exact layout:
        User: ...
        Bot: ...
        (blank line)
    """
    parts = []
    for user, bot in history:
        parts.append(f"User: {user}\nBot: {bot}\n\n")
    parts.append(f"User: {message}\nBot:")
    prompt = "".join(parts)

    ids = tokenizer.encode(prompt)
    if len(ids) > block_size:
        ids = ids[-block_size:]
        text = tokenizer.decode(ids)
        cut = text.find("User: ")
        if cut > 0:
            text = text[cut:]
            ids = tokenizer.encode(text)
    return torch.tensor([ids], dtype=torch.long)


def clean_reply(text):
    """Drop anything after the end of the bot turn."""
    for marker in STOP_MARKERS:
        if marker in text:
            text = text.split(marker)[0]
    return text.strip()
