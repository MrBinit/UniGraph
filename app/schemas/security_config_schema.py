from pydantic import BaseModel, Field


class SecurityConfig(BaseModel):
    auth_enabled: bool = True
    jwt_secret: str = Field(min_length=16)
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "ai-system"
    jwt_exp_minutes: int = Field(default=60, ge=1, le=1440)
    admin_roles: list[str] = Field(default_factory=lambda: ["admin"])
