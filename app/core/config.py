from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    data_dir: str = "data"
    uploads_dir: str = "data/uploads"
    models_dir: str = "data/models"
    chroma_dir: str = "data/chroma_db"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
