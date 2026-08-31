# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal demo attention backend for learning how to plug a custom backend
into vLLM's ``AttentionBackendEnum`` registry.

Unlike the tokenizer/renderer demos, attention backends require a real
kernel implementation (KV cache layout, metadata builder, etc.), so this
delegates all actual computation to ``CPUAttentionBackend`` and only adds a
logging hook around ``forward()``.

Real third-party backends register themselves the same way, e.g. from a
plugin's entrypoint:

    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    import vllm.v1.attention.backends.demo_attn  # noqa: F401  (registers on import)

    config = AttentionConfig(backend=AttentionBackendEnum.CUSTOM)
"""
import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.attention.backends.cpu_attn import (
    CPUAttentionBackend,
    CPUAttentionBackendImpl,
    CPUAttentionMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

logger = init_logger(__name__)


class DemoAttentionBackendImpl(CPUAttentionBackendImpl):
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CPUAttentionMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logger.info("[demo attn] forward() called: query.shape=%s", query.shape)
        return super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )


@register_backend(AttentionBackendEnum.CUSTOM)
class DemoAttentionBackend(CPUAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "DEMO_ATTN"

    @staticmethod
    def get_impl_cls() -> type["DemoAttentionBackendImpl"]:
        return DemoAttentionBackendImpl
