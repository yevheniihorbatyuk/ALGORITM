from collections import OrderedDict
from typing import Any, Optional
from datetime import datetime, timedelta

class LRUCache:
    def __init__(self, capacity: int, ttl_seconds: int = 3600):
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, tuple[Any, datetime]] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
            
        value, timestamp = self.cache[key]
        if (datetime.now() - timestamp).total_seconds() > self.ttl_seconds:
            del self.cache[key]
            return None
            
        self.cache.move_to_end(key)
        return value

    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = (value, datetime.now())
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)