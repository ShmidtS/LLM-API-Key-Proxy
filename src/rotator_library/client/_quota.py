from typing import Any, Dict, Optional

from ..quota_reporter import QuotaReporter


class QuotaMixin:
    @property
    def quota_reporter(self):
        if not hasattr(self, "_quota_reporter_instance"):
            self._quota_reporter_instance = QuotaReporter(
                self.usage_manager,
                self._provider_plugins,
                self._provider_instances,
                self.all_credentials,
            )
        return self._quota_reporter_instance

    async def get_quota_stats(
        self,
        provider_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self.quota_reporter.get_quota_stats(provider_filter)

    async def reload_usage_from_disk(self) -> None:
        """
        Force reload usage data from disk.

        Useful when wanting fresh stats without making external API calls.
        """
        await self.usage_manager.reload_from_disk()

    async def force_refresh_quota(
        self,
        provider: Optional[str] = None,
        credential: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self.quota_reporter.force_refresh_quota(provider, credential)