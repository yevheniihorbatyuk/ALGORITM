from dataclasses import dataclass
from sqlite3.dbapi2 import Timestamp

@dataclass
class UsageEvent:
    user_id:str  
    music_id:str 
    timestamp:str 
    duration_point:str 
    comtleted:str 
    device:str 