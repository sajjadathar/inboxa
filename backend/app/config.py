"""All settings come from the .env file, so switching AI model is a one-line change."""
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str

    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768  # size of each embedding vector; must match the database column
    store_name: str = "Luma Beauty"  # used to sign replies
    ollama_base_url: str = "http://ollama:11434"

    gemini_api_key: str = Field(
        default="", validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY")
    )
    groq_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""


settings = Settings()
