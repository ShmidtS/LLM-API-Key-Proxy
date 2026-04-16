# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""Batch embedding request aggregation. Collects individual embedding requests and merges them into batched API calls for improved throughput."""

import asyncio
from typing import List, Dict, Any, Tuple
import time
from rotator_library import RotatingClient

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
                # No items arrived during the timeout window — yield to event loop
                await asyncio.sleep(0)
                continue

            try:
                # Assume all requests in a batch use the same model and other settings
                model = batch[0]["model"]
                inputs = [item["input"][0] for item in batch] # Extract single string input

                batched_request = {
                    "model": model,
                    "input": inputs
                }
                
                # Pass through any other relevant parameters from the first request
                for key in ["input_type", "dimensions", "user"]:
                    if key in batch[0]:
                        batched_request[key] = batch[0][key]

                response = await self.client.aembedding(**batched_request)
                
                # Distribute results back to the original requesters
                for i, future in enumerate(futures):
                    if i >= len(response.data):
                        future.set_exception(IndexError(f"Batch response has {len(response.data)} items but batch has {len(futures)} requests"))
                        continue
                    # Create a new response object for each item in the batch
                    single_response_data = {
                        "object": response.object,
                        "model": response.model,
                        "data": [response.data[i]],
                        "usage": response.usage # Usage is for the whole batch
                    }
                    future.set_result(single_response_data)

            except asyncio.CancelledError:
                for future in futures:
                    if not future.done():
                        future.cancel()
                raise

            except Exception as e:
                for future in futures:
                    future.set_exception(e)

    async def _gather_batch(self) -> Tuple[List[Dict[str, Any]], List[asyncio.Future]]:
        """Collect requests from the queue until batch_size or timeout is reached. Returns: Tuple of (batch request data list, corresponding futures list)."""
        batch = []
        futures = []
        start_time = time.time()

        while len(batch) < self.batch_size and (time.time() - start_time) < self.timeout:
            try:
                # Wait for an item with a timeout
                timeout = self.timeout - (time.time() - start_time)
                if timeout <= 0:
                    break
                request, future = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                batch.append(request)
                futures.append(future)
            except asyncio.TimeoutError:
                break
        
        return batch, futures

    async def stop(self):
        """Cancel the background worker and drain pending requests."""
        self.worker_task.cancel()
        try:
            await self.worker_task
        except asyncio.CancelledError:
            pass
        # Drain any pending futures so callers don't hang forever
        cancelled_exc = asyncio.CancelledError("EmbeddingBatcher stopped")
        while not self.queue.empty():
            try:
                _, future = self.queue.get_nowait()
                if not future.done():
                    future.set_exception(cancelled_exc)
            except asyncio.QueueEmpty:
                break
        # CancelledError is expected during deliberate shutdown;
        # do not re-raise so callers can continue cleanup (close_doh_client, close_http_pool, etc.)