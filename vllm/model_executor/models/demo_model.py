# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal demo model for learning how to plug a custom architecture into
vLLM's ``ModelRegistry``.

Unlike the tokenizer/renderer/attention-backend demos, a model still needs a
full, working forward pass, so this delegates all actual computation to
``GPT2LMHeadModel`` and only adds a logging hook around ``forward()``.

Real third-party models register themselves the same way, e.g. from a
plugin's entrypoint:

    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "DemoForCausalLM", "vllm.model_executor.models.demo_model:DemoForCausalLM"
    )
"""
from collections.abc import Iterable

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from .gpt2 import GPT2LMHeadModel

logger = init_logger(__name__)


class DemoForCausalLM(GPT2LMHeadModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        logger.info("[demo model] DemoForCausalLM initialized")

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        logger.info(
            "[demo model] forward() called: input_ids.shape=%s",
            None if input_ids is None else input_ids.shape,
        )
        return super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        logger.info("[demo model] load_weights() called")
        return super().load_weights(weights)
