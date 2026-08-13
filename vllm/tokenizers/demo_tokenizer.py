# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal demo tokenizer for learning how to plug a custom tokenizer into
vLLM's ``TokenizerRegistry``.

It delegates all real tokenization to the standard HF tokenizer and only
adds a thin logging wrapper around ``encode``/``decode``. A real
implementation (see ``deepseek_v32.py`` or ``mistral.py`` for examples)
would instead swap in a bespoke tokenizer/backend here.

Try it with:
    vllm serve <model> --tokenizer-mode demo
"""
import copy

from vllm.logger import init_logger

from .hf import CachedHfTokenizer, HfTokenizer
from .protocol import TokenizerLike

logger = init_logger(__name__)


def _wrap_demo_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    demo_tokenizer = copy.copy(tokenizer)

    class _DemoTokenizer(tokenizer.__class__):  # type: ignore
        def encode(self, text, *args, **kwargs):
            logger.info("[demo tokenizer] encode() called on: %r", text)
            return super().encode(text, *args, **kwargs)

        def decode(self, ids, *args, **kwargs):
            result = super().decode(ids, *args, **kwargs)
            logger.info("[demo tokenizer] decode() called -> %r", result)
            return result

        def __reduce__(self):
            return _wrap_demo_tokenizer, (tokenizer,)

    _DemoTokenizer.__name__ = f"Demo{tokenizer.__class__.__name__}"
    demo_tokenizer.__class__ = _DemoTokenizer
    return demo_tokenizer


class DemoTokenizer(TokenizerLike):
    """Example custom tokenizer registered under ``tokenizer_mode="demo"``."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> HfTokenizer:
        tokenizer = CachedHfTokenizer.from_pretrained(*args, **kwargs)
        return _wrap_demo_tokenizer(tokenizer)
