import redis
import json
from typing import Any, Optional
from datetime import timedelta

class RedisCache:
    def __init__(self, capacity: int = 1000, ttl_seconds: int = 3600):
        self.redis = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        data = self.redis.get(key)
        return json.loads(data) if data else None

    def put(self, key: str, value: Any) -> None:
        self.redis.setex(key, timedelta(seconds=self.ttl), json.dumps(value))