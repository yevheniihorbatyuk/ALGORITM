from typing import Dict, Any
import asyncio
from openmeteo_requests import Client
import requests_cache
from retry_requests import retry
from datetime import datetime
from .base import BaseExternalAPI


class WeatherProvider(BaseExternalAPI):
    def __init__(self):
        super().__init__("WeatherProvider", rate_limit=10, time_window=1)
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.client = Client(session=retry_session)

    async def search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        coordinates = {
            "Paris": {"latitude": 48.8566, "longitude": 2.3522},
            "London": {"latitude": 51.5074, "longitude": -0.1278},
            "New York": {"latitude": 40.7128, "longitude": -74.0060}
        }
        
        location = params.get("destination", "Paris")
        coords = coordinates.get(location, coordinates["Paris"])
        
        params = {
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
            "hourly": ["temperature_2m", "precipitation_probability"],
            "timezone": "auto"
        }
        
        responses = self.client.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
        response = responses[0]
        
        hourly = response.Hourly()
        time = [datetime.fromisoformat(t).strftime("%Y-%m-%d %H:%M") 
               for t in hourly.Time()]
        temp = hourly.Variables(0).ValuesAsNumpy()
        precip = hourly.Variables(1).ValuesAsNumpy()

        forecast = []
        for i in range(len(time)):
            forecast.append({
                "time": time[i],
                "temperature": float(temp[i]),
                "precipitation_probability": float(precip[i])
            })

        return {
            "provider": self.name,
            "location": location,
            "coordinates": coords,
            "forecast": forecast[:24]  # Return first 24 hours
        }


