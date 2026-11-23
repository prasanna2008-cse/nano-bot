"""Character-level tokenizer: maps every character to an integer id."""

import json


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    @classmethod
    def from_text(cls, text):
        """Build the vocabulary from all unique characters in the text."""
        return cls(sorted(set(text)))

    def encode(self, text):
        # Unknown characters are skipped instead of raising, so typing an
        # emoji into the demo does not crash the server.
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["chars"])
