from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Dict, Any
import asyncio
from datetime import datetime

from .lru_cache import LRUCache
from .redis_cache import RedisCache
from .rate_limit import SlidingWindowRateLimiter
from .base import BaseExternalAPI
from .provider_1 import WeatherProvider
from .provider_2 import ZippoProvider
from config import get_settings

router = APIRouter()
settings = get_settings()


# Place for init cache and limiter

cache = RedisCache(settings.CACHE_SIZE, settings.CACHE_TTL)
rate_limiter = SlidingWindowRateLimiter(
    window_size=settings.RATE_LIMIT_WINDOW,
    max_requests=settings.RATE_LIMIT_REQUESTS
)


# providers

weather_provider, zippo_provider = [
    WeatherProvider(),
    ZippoProvider()
]


@router.get('/weather/{country}')
async def get_weather(
    request: Request,
    country: str,

):
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests"
        )

    cache_key = f"weather:{country}"
    cached_result = cache.get(cache_key)
    if cached_result:
        return {"source": "cache", "results": cached_result}

    try:
        result = await weather_provider.execute_request({"destination": country})
        cache.put(cache_key, result)
        return {"source": "provider", "results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.get('/zipcode/{country}/{code}')
async def get_zipcode(
    request: Request,
    country: str,
    code: str
):
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests"
        )

    cache_key = f"zipcode:{country}:{code}"
    cached_result = cache.get(cache_key)
    if cached_result:
        return {"source": "cache", "results": cached_result}

    try:
        result = await zippo_provider.execute_request({
            "country": country,
            "zipcode": code
        })
        cache.put(cache_key, result)
        return {"source": "provider", "results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





