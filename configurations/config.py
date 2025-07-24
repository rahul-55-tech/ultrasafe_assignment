from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Application
    app_name: str = "UltraSafe Data Processing and Analysis System"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Pinecone Vector Database
    pinecone_api_key: str
    pinecone_index_name: str = "ultrasafe-data-index"
    

    # OpenAI (for embeddings and LLM)
    openai_api_key: Optional[str] = None
    
    # File Upload
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_file_types: list = [".csv", ".xlsx", ".json", ".parquet"]
    upload_dir: str = "uploads"
    
    # RAG Settings
    embedding_model: str = "text-embedding-ada-002"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    similarity_threshold: float = 0.7
    
    # Agent Settings
    max_agents: int = 5
    agent_timeout: int = 300  # in sec

    MAX_FILE_SIZE: int = 104857600
    UPLOAD_DIR: str = "uploads"
    EXPORT_DIR: str = "exports"
    STATIC_DIR: str = "static"
    LLM_MODEL: str = "gpt-3.5-turbo"
    USF_OPENAPIKEY: str = "your ultrasafe api key"
    CONFIGURED_LLM: str = "ultrasafe"

    kb_file_path:str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../data/statistical_methods.json"))
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Create settings instance
settings = Settings()
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs("db", exist_ok=True)