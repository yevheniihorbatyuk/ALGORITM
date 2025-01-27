from abc import ABC, abstractmethod
from typing import Optional, Deque
import redis
from datetime import datetime, timedelta
from collections import deque

class BaseRateLimiter(ABC):
    @abstractmethod
    def is_allowed(self, key: str) -> bool:
        pass

class InMemoryRateLimiter(BaseRateLimiter):
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests: dict[str, deque[datetime]] = {}

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

class RedisRateLimiter(BaseRateLimiter):
    def __init__(self, redis_client: redis.Redis, window_size: int, max_requests: int):
        self.redis = redis_client
        self.window_size = window_size
        self.max_requests = max_requests

    def is_allowed(self, key: str) -> bool:
        pipe = self.redis.pipeline()
        now = datetime.now().timestamp()
        window_start = now - self.window_size
        
        # Remove old requests
        pipe.zremrangebyscore(key, 0, window_start)
        # Count requests in window
        pipe.zcard(key)
        # Add new request
        pipe.zadd(key, {str(now): now})
        # Set expiry
        pipe.expire(key, self.window_size)
        
        results = pipe.execute()
        request_count = results[1]
        
        return request_count < self.max_requests

class TokenBucketRateLimiter(BaseRateLimiter):
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens: dict[str, float] = {}
        self.last_update: dict[str, datetime] = {}

    def is_allowed(self, key: str) -> bool:
        now = datetime.now()
        
        if key not in self.tokens:
            self.tokens[key] = self.capacity
            self.last_update[key] = now
            return True
            
        # Refill tokens
        elapsed = (now - self.last_update[key]).total_seconds()
        self.tokens[key] = min(
            self.capacity,
            self.tokens[key] + elapsed * self.refill_rate
        )
        self.last_update[key] = now
        
        if self.tokens[key] >= 1:
            self.tokens[key] -= 1
            return True
        return False

class SlidingWindowRateLimiter(BaseRateLimiter):
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