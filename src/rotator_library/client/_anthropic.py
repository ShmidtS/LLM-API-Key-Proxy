from typing import TYPE_CHECKING, Any, Callable, Optional

from ..anthropic_adapter import AnthropicAdapter
from ..utils.model_utils import extract_provider_from_model

if TYPE_CHECKING:
    from ..anthropic_compat.models import (
        AnthropicCountTokensRequest,
        AnthropicMessagesRequest,
    )


class AnthropicCompatibilityMixin:
    @property
    def anthropic_adapter(self):
        if not hasattr(self, "_anthropic_adapter_instance"):
            self._anthropic_adapter_instance = AnthropicAdapter(
                self.acompletion,
                self.token_count,
                extract_provider_from_model,
                self.all_credentials,
                self.enable_request_logging,
            )
        return self._anthropic_adapter_instance

    async def anthropic_messages(
        self,
        request: "AnthropicMessagesRequest",
        raw_request: Optional[Any] = None,
        pre_request_callback: Optional[Callable] = None,
        raw_body_data: Optional[dict] = None,
    ) -> Any:
        return await self.anthropic_adapter.anthropic_messages(
            request, raw_request, pre_request_callback, raw_body_data
        )

    async def anthropic_count_tokens(
        self,
        request: "AnthropicCountTokensRequest",
    ) -> dict:
        return await self.anthropic_adapter.anthropic_count_tokens(request)