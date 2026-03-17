"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings read from environment variables or .env file."""

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    max_execution_time: int = Field(default=5, alias="MAX_EXECUTION_TIME")
    max_memory_mb: int = Field(default=256, alias="MAX_MEMORY_MB")
    debug: bool = Field(default=False, alias="DEBUG")

    model_config = {"env_file": ".env", "populate_by_name": True}


settings = Settings()
