from pydantic import BaseModel


class AppConfig(BaseModel):
    name: str
    log_level: str
