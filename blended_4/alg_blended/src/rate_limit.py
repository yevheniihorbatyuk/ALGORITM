from datetime import datetime, timedelta
from collections import deque
from typing import Deque, Tuple

class SlidingWindowRateLimiter:
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests: dict[str, Deque[datetime]] = {}

    def is_allowed(self, key: str) -> bool:
        if key not in self.requests:
            self.requests[key] = deque()

        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_size)

        while self.requests[key] and self.requests[key][0] < window_start:
            self.requests[key].popleft()

        if len(self.requests[key]) < self.max_requests:
            self.requests[key].append(now)
            return True

        return False