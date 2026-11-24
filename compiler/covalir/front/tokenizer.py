# Copyright 2026 CovalIR Contributors
# Licensed under the Apache License, Version 2.0
"""Zero-dependency reader/decoder for Karpathy llama2.c `tokenizer.bin`."""

from __future__ import annotations

import pathlib
import struct


class Llama2Tokenizer:
    """Loads vocab strings and scores from Karpathy's `tokenizer.bin`."""

    def __init__(self, path: pathlib.Path, vocab_size: int = 32000) -> None:
        self.vocab: list[str] = []
        self.scores: list[float] = []
        with path.open("rb") as f:
            self.max_token_length = struct.unpack("<i", f.read(4))[0]
            for _ in range(vocab_size):
                score = struct.unpack("<f", f.read(4))[0]
                tlen = struct.unpack("<i", f.read(4))[0]
                piece = f.read(tlen).decode("utf-8", errors="replace")
                self.scores.append(score)
                self.vocab.append(piece)

    def decode_token(self, tok_id: int) -> str:
        piece = self.vocab[tok_id]
        if piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6:
            return bytes([int(piece[3:5], 16)]).decode("utf-8", errors="replace")
        return piece

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.decode_token(t) for t in token_ids)
