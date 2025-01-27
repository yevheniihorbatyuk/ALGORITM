from fastapi import FastAPI
from src.endpoints import router
from config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="Service Aggregator with Rate Limiting and Caching"
)

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8899)