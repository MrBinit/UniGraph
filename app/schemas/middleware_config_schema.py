from pydantic import BaseModel, Field


class MiddlewareConfig(BaseModel):
    timeout_seconds: int = Field(default=35, ge=1, le=120)
    max_in_flight_requests: int = Field(default=200, ge=1, le=5000)
    rate_limit_requests: int = Field(default=120, ge=1, le=100000)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)
    enable_request_logging: bool = True
    enable_rate_limit: bool = True
    enable_timeout: bool = True
    enable_backpressure: bool = True
    enable_route_matching: bool = True
