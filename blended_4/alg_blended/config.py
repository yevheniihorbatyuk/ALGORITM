from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Literal
from src.caches import (
    BaseCache,
    RedisCache,
    MemcachedCache,
    MultiLevelCache
)
from src.rate_limit import (
    InMemoryRateLimiter,
    RedisRateLimiter,
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter
)

class Settings(BaseSettings):
    # Cache settings
    CACHE_TYPE: Literal["memory", "redis", "memcached", "multi"] = "memory"
    CACHE_STRATEGY: Literal["lru", "ttl"] = "lru"
    CACHE_SIZE: int = 1000
    CACHE_TTL: int = 3600
    
    # Redis settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # Memcached settings
    MEMCACHED_SERVERS: list[str] = ["localhost:11211"]
    
    # Rate limit settings
    RATE_LIMIT_TYPE: Literal["memory", "redis", "token"] = "memory"
    RATE_LIMIT_WINDOW: int = 3600
    RATE_LIMIT_REQUESTS: int = 1000
    TOKEN_BUCKET_CAPACITY: int = 100
    TOKEN_BUCKET_REFILL_RATE: float = 1.0
    
    # API Provider settings
    WEATHER_API_KEY: str = ""
    NEWS_API_KEY: str = ""
    TRANSLATION_API_KEY: str = ""

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()

def get_cache() -> BaseCache:
    settings = get_settings()
    
    if settings.CACHE_TYPE == "redis":
        return RedisCache(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            ttl_seconds=settings.CACHE_TTL
        )
    elif settings.CACHE_TYPE == "memcached":
        return MemcachedCache(
            servers=settings.MEMCACHED_SERVERS,
            ttl_seconds=settings.CACHE_TTL
        )
    elif settings.CACHE_TYPE == "multi":
        return MultiLevelCache([
            MemoryCache(settings.CACHE_STRATEGY, settings.CACHE_SIZE, settings.CACHE_TTL),
            RedisCache(ttl_seconds=settings.CACHE_TTL)
        ])
    else:
        return MemoryCache(
            strategy=settings.CACHE_STRATEGY,
            capacity=settings.CACHE_SIZE,
            ttl_seconds=settings.CACHE_TTL
        )

def get_rate_limiter() -> BaseRateLimiter:
    settings = get_settings()
    
    if settings.RATE_LIMIT_TYPE == "redis":
        redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB
        )
        return RedisRateLimiter(
            redis_client,
            settings.RATE_LIMIT_WINDOW,
            settings.RATE_LIMIT_REQUESTS
        )
    elif settings.RATE_LIMIT_TYPE == "token":