# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""Batch embedding request aggregation. Collects individual embedding requests and merges them into batched API calls for improved throughput."""

import asyncio
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import logging
import time
from rotator_library import RotatingClient

logger = logging.getLogger(__name__)

class EmbeddingBatcher:
    """Aggregates individual embedding requests into batched calls. Collects requests via a queue, groups them by model, and sends batched API requests to reduce overhead."""

    def __init__(self, client: RotatingClient, batch_size: int = 64, timeout: float = 0.1):
        """Initialize the batcher. Args: client: RotatingClient instance for making API calls. batch_size: Maximum number of requests per batch. timeout: Maximum seconds to wait before sending a partial batch."""
        self.client = client
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue = asyncio.Queue()
        self.worker_task = asyncio.create_task(self._batch_worker())

    async def add_request(self, request_data: Dict[str, Any]) -> Any:
        """Submit a single embedding request and wait for its result. Args: request_data: Dict with model, input, and optional parameters. Returns: The embedding response for this specific request."""
        future = asyncio.Future()
        await self.queue.put((request_data, future))
        return await future

    async def _batch_worker(self):
        """Background coroutine that gathers requests and sends batched API calls."""
        while True:
            batch, futures = await self._gather_batch()
            if not batch:
                continue

            grouped_batches = defaultdict(lambda: {"batch": [], "futures": []})
            for request, future in zip(batch, futures):
                batch_key = (
                    request.get("model"),
                    request.get("input_type"),
                    request.get("dimensions"),
                    request.get("user"),
                )
                grouped_batches[batch_key]["batch"].append(request)
                grouped_batches[batch_key]["futures"].append(future)

            for grouped in grouped_batches.values():
                grouped_batch = grouped["batch"]
                grouped_futures = grouped["futures"]

                try:
                    model = grouped_batch[0]["model"]
                    first_input = grouped_batch[0]["input"]
                    if isinstance(first_input, list):
                        if len(grouped_batch) != 1:
                            raise ValueError("Embedding batch requests with list input cannot be merged")
                        inputs = first_input
                    else:
                        inputs = [item["input"] for item in grouped_batch]

                    batched_request = {
                        "model": model,
                        "input": inputs,
                    }

                    for key in ["input_type", "dimensions", "user"]:
                        if key in grouped_batch[0]:
                            batched_request[key] = grouped_batch[0][key]

                    response = await self.client.aembedding(**batched_request)

                    for i, future in enumerate(grouped_futures):
                        if future.done():
                            continue
                        if i >= len(response.data):
                            future.set_exception(
                                IndexError(
                                    f"Batch response has {len(response.data)} items but batch has {len(grouped_futures)} requests"
                                )
                            )
                            continue
                        single_response_data = {
                            "object": response.object,
                            "model": response.model,
                            "data": [response.data[i]],
                            "usage": None,
                        }
                        future.set_result(single_response_data)

                except asyncio.CancelledError:
                    for future in grouped_futures:
                        if not future.done():
                            future.cancel()
                    raise

                except Exception as e:
                    for future in grouped_futures:
                        if not future.done():
                            future.set_exception(e)

    async def _gather_batch(self) -> Tuple[List[Dict[str, Any]], List[asyncio.Future]]:
        """Collect requests from the queue until batch_size or timeout is reached."""
        batch = []
        futures = []
        start_time = time.monotonic()

        try:
            while len(batch) < self.batch_size:
                remaining = self.timeout - (time.monotonic() - start_time)
                if remaining <= 0:
                    break
                request, future = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                batch.append(request)
                futures.append(future)
        except asyncio.TimeoutError:
            logger.debug("Batch collection timed out, returning partial batch", exc_info=True)

        return batch, futures

    async def stop(self):
        """Cancel the background worker and drain pending requests."""
        self.worker_task.cancel()
        try:
            await self.worker_task
        except asyncio.CancelledError:
            raise
        finally:
            # Drain any pending futures so callers don't hang forever
            cancelled_exc = asyncio.CancelledError("EmbeddingBatcher stopped")
            while not self.queue.empty():
                try:
                    _, future = self.queue.get_nowait()
                    if not future.done():
                        future.set_exception(cancelled_exc)
                except asyncio.QueueEmpty:
                    break
