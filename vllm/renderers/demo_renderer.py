# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal demo renderer for learning how to plug a custom renderer into
vLLM's ``RendererRegistry``.

It subclasses ``HfRenderer`` and only adds logging around the two render
hooks; all actual chat-template/multimodal handling is delegated to the
parent class. A real implementation (see ``deepseek_v32.py`` or
``mistral.py`` for examples) would instead customize how messages are
turned into a prompt.

Try it with:
    vllm serve <model> --tokenizer-mode demo
"""
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
)
from vllm.logger import init_logger

from .hf import HfRenderer
from .inputs import DictPrompt
from .params import ChatParams

logger = init_logger(__name__)


class DemoRenderer(HfRenderer):
    """Example custom renderer registered under ``renderer_mode="demo"``."""

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        logger.info(
            "[demo renderer] render_messages() called with %d messages",
            len(messages),
        )
        return super().render_messages(messages, params)

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        logger.info(
            "[demo renderer] render_messages_async() called with %d messages",
            len(messages),
        )
        return await super().render_messages_async(messages, params)
