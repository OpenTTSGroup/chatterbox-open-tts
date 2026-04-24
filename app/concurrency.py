from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import HTTPException

from app.schemas import ConcurrencySnapshot


class ConcurrencyLimiter:
    """Asyncio semaphore + bounded queue + optional wait timeout.

    Use as ``async with limiter.acquire(): ...``. For streaming endpoints wrap
    the whole generator so the permit is held until the stream ends.
    """

    def __init__(
        self,
        max_concurrency: int,
        max_queue_size: int,
        queue_timeout: float,
    ) -> None:
        self._max = max(1, int(max_concurrency))
        self._max_queue = max(0, int(max_queue_size))
        self._timeout = max(0.0, float(queue_timeout))
        self._sem = asyncio.Semaphore(self._max)
        self._queued = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        async with self._lock:
            if self._max_queue > 0 and self._queued >= self._max_queue:
                raise HTTPException(status_code=503, detail="queue full")
            self._queued += 1

        try:
            if self._timeout > 0:
                try:
                    await asyncio.wait_for(self._sem.acquire(), timeout=self._timeout)
                except asyncio.TimeoutError as exc:
                    raise HTTPException(
                        status_code=503, detail="queue timeout"
                    ) from exc
            else:
                await self._sem.acquire()
        finally:
            async with self._lock:
                self._queued -= 1

        try:
            yield
        finally:
            self._sem.release()

    def snapshot(self) -> ConcurrencySnapshot:
        active = self._max - self._sem._value  # pragma: no cover - observation only
        if active < 0:
            active = 0
        return ConcurrencySnapshot(max=self._max, active=active, queued=self._queued)
