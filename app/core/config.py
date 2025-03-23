from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    EMBEDDING_MODEL: str = "C:/Users/roame/project/rag-vue/model_cache"  # 指向本地模型路径
    LLM_MODEL: str = "openai"
    OPENAI_API_KEY: str = None
    FAISS_INDEX_PATH: str = "data/faiss_index"  # 新增向量存储路径
    DATA_DIR: str = "data/uploads"              # 文档存储目录

    class Config:
        env_file = ".env"

settings = Settings()