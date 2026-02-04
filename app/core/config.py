from pydantic import BaseSettings


class Settings(BaseSettings):
    data_dir: str = "data"
    uploads_dir: str = "data/uploads"
    models_dir: str = "data/models"
    chroma_dir: str = "data/chroma_db"

    class Config:
        env_file = ".env"


settings = Settings()
