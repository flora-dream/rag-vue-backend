from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    EMBEDDING_MODEL: str = "C:/Users/roame/project/rag-vue/model_cache"  # 指向本地模型路径
    LLM_MODEL: str = "openai"
    OPENAI_API_KEY: str = None
    DATA_DIR: str = "./data"

    class Config:
        env_file = ".env"

settings = Settings()