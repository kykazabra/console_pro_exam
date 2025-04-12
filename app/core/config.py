from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    API_KEY: str

    OPENAI_API_KEY: str
    OPENAI_API_URL: str
    OPENAI_EMBEDDING_MODEL: str
    OPENAI_LLM_MODEL: str

    SQLITE_PATH: str
    CHROMA_PATH: str

    DOC_URL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.SQLITE_PATH = os.path.abspath(os.getcwd()) + '/' + self.SQLITE_PATH
        self.CHROMA_PATH = os.path.abspath(os.getcwd()) + '/' + self.CHROMA_PATH


settings = Settings()