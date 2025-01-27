from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "Service Aggregator"
    RATE_LIMIT_REQUESTS: int = 1000
    RATE_LIMIT_WINDOW: int = 60  # seconds
    CACHE_SIZE: int = 10
    CACHE_TTL: int = 10  # seconds
    QUEUE_MAX_SIZE: int = 100

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()