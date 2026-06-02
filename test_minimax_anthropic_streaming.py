# SPDX-License-Identifier: LGPL-3.0-only

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rotator_library.anthropic_compat.streaming_fast import anthropic_streaming_wrapper  # noqa: E402
from rotator_library.anthropic_compat.translator import openai_to_anthropic_response  # noqa: E402
from rotator_library.utils.json_utils import STREAM_DONE, json_loads  # noqa: E402


def _parse_sse_events(chunks):
    events = []
    event_name = None
    data_lines = []
    for line in "".join(chunks).splitlines():
        if line.startswith("event: "):
            event_name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            data_lines.append(line.removeprefix("data: "))
        elif not line and data_lines:
            events.append((event_name, json_loads("\n".join(data_lines))))
            event_name = None
            data_lines = []
    return events


def _assert_content_block_delta_types(test_case, events):
    block_types = {}
    delta_to_block_type = {
        "text_delta": "text",
        "thinking_delta": "thinking",
        "input_json_delta": "tool_use",
    }

    for _event_name, payload in events:
        payload_type = payload.get("type")
        if payload_type == "content_block_start":
            index = payload["index"]
            test_case.assertNotIn(index, block_types)
            block_types[index] = payload["content_block"]["type"]
        elif payload_type == "content_block_delta":
            index = payload["index"]
            delta_type = payload["delta"]["type"]
            test_case.assertEqual(block_types[index], delta_to_block_type[delta_type])


class MinimaxAnthropicStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_minimax_think_tags_become_thinking_block(self):
        async def openai_stream():
            yield {"choices": [{"index": 0, "delta": {"content": "<thi"}}]}
            yield {"choices": [{"index": 0, "delta": {"content": "nk>hidden</thi"}}]}
            yield {"choices": [{"index": 0, "delta": {"content": "nk>Visible"}}]}
            yield STREAM_DONE

        chunks = [
            event
            async for event in anthropic_streaming_wrapper(
                openai_stream=openai_stream(),
                original_model="minimax/MiniMax-M3",
                request_id="msg_test",
            )
        ]

        joined = "".join(chunks)
        self.assertNotIn("<think>", joined)
        self.assertNotIn("</think>", joined)

        events = _parse_sse_events(chunks)
        _assert_content_block_delta_types(self, events)

        starts = [
            payload["content_block"]["type"]
            for _event_name, payload in events
            if payload.get("type") == "content_block_start"
        ]
        self.assertEqual(starts, ["thinking", "text"])

    async def test_late_reasoning_content_uses_new_block_index(self):
        async def openai_stream():
            yield {"choices": [{"index": 0, "delta": {"content": "Visible"}}]}
            yield {"choices": [{"index": 0, "delta": {"reasoning_content": "hidden"}}]}
            yield {"choices": [{"index": 0, "delta": {"content": " again"}}]}
            yield STREAM_DONE

        chunks = [
            event
            async for event in anthropic_streaming_wrapper(
                openai_stream=openai_stream(),
                original_model="minimax/MiniMax-M3",
                request_id="msg_test",
            )
        ]

        events = _parse_sse_events(chunks)
        _assert_content_block_delta_types(self, events)

        starts = [
            (payload["index"], payload["content_block"]["type"])
            for _event_name, payload in events
            if payload.get("type") == "content_block_start"
        ]
        self.assertEqual(starts, [(0, "text"), (1, "thinking"), (2, "text")])

    def test_minimax_non_streaming_think_tags_become_blocks(self):
        response = openai_to_anthropic_response(
            {
                "id": "chatcmpl_test",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "<think>hidden</think>Visible",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                },
            },
            "minimax/MiniMax-M3",
        )

        self.assertEqual(
            response["content"],
            [
                {"type": "thinking", "thinking": "hidden"},
                {"type": "text", "text": "Visible"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
