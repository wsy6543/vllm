# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import ToolParser

logger = init_logger(__name__)


class MyModelToolParser(ToolParser):
    """
    Demo tool call parser for models that emit a single bare JSON object
    as the tool call, with no wrapping tags:

        {"name": "get_weather", "arguments": {"city": "Beijing"}}

    Used when --tool-call-parser my_model is set.
    """

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        stripped = model_output.strip()
        if not stripped.startswith("{"):
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            data = json.loads(stripped)
            if "name" not in data:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls = [
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=data["name"],
                        arguments=json.dumps(
                            data.get("arguments", {}), ensure_ascii=False
                        ),
                    ),
                )
            ]
            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=None
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        stripped = current_text.strip()
        if not stripped.startswith("{"):
            return DeltaMessage(content=delta_text)

        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            # Not enough tokens yet to form valid JSON.
            return None

        if "name" not in data:
            return None

        if not self.current_tool_name_sent:
            self.current_tool_name_sent = True
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0,
                        type="function",
                        id=make_tool_call_id(),
                        function=DeltaFunctionCall(name=data["name"]).model_dump(
                            exclude_none=True
                        ),
                    )
                ]
            )

        if not self.streamed_args_for_tool:
            self.streamed_args_for_tool.append("")

        arguments = json.dumps(data.get("arguments", {}), ensure_ascii=False)
        sent = self.streamed_args_for_tool[0]
        if not arguments.startswith(sent):
            return None
        diff = arguments[len(sent) :]
        if not diff:
            return None

        self.streamed_args_for_tool[0] = arguments
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    function=DeltaFunctionCall(arguments=diff).model_dump(
                        exclude_none=True
                    ),
                )
            ]
        )
