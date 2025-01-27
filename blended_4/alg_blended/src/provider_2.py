from typing import Dict, Any
import asyncio
from openmeteo_requests import Client
import requests_cache
from retry_requests import retry
from datetime import datetime
from .base import BaseExternalAPI

class ZippoProvider(BaseExternalAPI):
   def __init__(self):
       super().__init__("ZippoProvider", rate_limit=10, time_window=1)
       self.base_url = "https://api.zippopotam.us"

   async def search(self, params: Dict[str, Any]) -> Dict[str, Any]:
       import requests
       zipcode = params.get("zipcode")
       country = params.get("country", "us")
       
       if not zipcode:
           raise ValueError("Zipcode is required")

       response = requests.get(f"{self.base_url}/{country}/{zipcode}")
       
       if response.status_code == 404:
           return {
               "provider": self.name,
               "found": False,
               "message": "ZIP code not found"
           }
       
       data = response.json()
       return {
           "provider": self.name,
           "found": True, 
           "country": data.get("country"),
           "country_abbreviation": data.get("country abbreviation"),
           "places": data.get("places", [])
       }
