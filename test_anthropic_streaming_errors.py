# SPDX-License-Identifier: LGPL-3.0-only

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rotator_library.anthropic_compat.streaming_fast import anthropic_streaming_wrapper  # noqa: E402
from rotator_library.utils.json_utils import STREAM_DONE, json_loads  # noqa: E402


class AnthropicStreamingErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_internal_error_chunk_becomes_anthropic_error_event(self):
        async def openai_stream():
            yield {
                "error": {
                    "message": "Provider 'baidu' has multiple overdue/billing-error accounts.",
                    "type": "proxy_account_billing_error",
                }
            }
            yield STREAM_DONE

        events = [
            event
            async for event in anthropic_streaming_wrapper(
                openai_stream=openai_stream(),
                original_model="baidu/glm-5.1",
                request_id="msg_test",
            )
        ]

        joined = "".join(events)
        self.assertIn("event: error", joined)
        self.assertNotIn("event: message_start", joined)
        self.assertNotIn("event: message_stop", joined)

        data_line = next(line for line in joined.splitlines() if line.startswith("data: "))
        payload = json_loads(data_line.removeprefix("data: "))
        self.assertEqual(payload["type"], "error")
        self.assertEqual(payload["error"]["type"], "api_error")
        self.assertIn("baidu", payload["error"]["message"])


if __name__ == "__main__":
    unittest.main()
