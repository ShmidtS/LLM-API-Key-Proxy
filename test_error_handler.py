# SPDX-License-Identifier: LGPL-3.0-only

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from litellm.exceptions import (  # noqa: E402
    APIError as LiteLLMAPIError,
    BadRequestError,
    InvalidRequestError,
)

from rotator_library.error_handler import classify_error, should_rotate_on_error  # noqa: E402


class ClassifyErrorTests(unittest.TestCase):
    def test_bad_request_invalid_iam_token_is_authentication(self):
        exc = BadRequestError(
            message="OpenAIException - invalid_iam_token",
            model="baidu/glm-5.1",
            llm_provider="baidu",
        )

        classified = classify_error(exc, provider="baidu")

        self.assertEqual(classified.error_type, "authentication")
        self.assertEqual(classified.status_code, 401)
        self.assertIs(classified.original_exception, exc)
        self.assertTrue(should_rotate_on_error(classified))

    def test_invalid_request_invalid_iam_token_is_authentication(self):
        exc = InvalidRequestError(
            message="OpenAIException - invalid iam token",
            model="baidu/glm-5.1",
            llm_provider="baidu",
        )

        classified = classify_error(exc, provider="baidu")

        self.assertEqual(classified.error_type, "authentication")
        self.assertEqual(classified.status_code, 401)
        self.assertIs(classified.original_exception, exc)
        self.assertTrue(should_rotate_on_error(classified))

    def test_ordinary_bad_request_remains_invalid_request(self):
        exc = BadRequestError(
            message="Unsupported parameter: response_format",
            model="baidu/glm-5.1",
            llm_provider="baidu",
        )

        classified = classify_error(exc, provider="baidu")

        self.assertEqual(classified.error_type, "invalid_request")
        self.assertEqual(classified.status_code, 400)
        self.assertFalse(should_rotate_on_error(classified))

    def test_account_overdue_api_error_is_quota_exceeded(self):
        exc = LiteLLMAPIError(
            status_code=403,
            message="OpenAIException - Access denied due to overdue account",
            llm_provider="baidu",
            model="baidu/glm-5.1",
        )

        classified = classify_error(exc, provider="baidu")

        self.assertEqual(classified.error_type, "quota_exceeded")
        self.assertEqual(classified.status_code, 403)
        self.assertEqual(classified.reason, "litellm_api_credits")


if __name__ == "__main__":
    unittest.main()
