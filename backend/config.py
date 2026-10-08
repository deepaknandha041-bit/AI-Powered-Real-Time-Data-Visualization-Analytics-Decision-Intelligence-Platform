import os

class Settings:
    PROJECT_NAME: str = "DataVista AI"
    API_PREFIX: str = "/api"
    UPLOAD_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
    DEMO_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data")
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: list[str] = [".csv", ".xlsx", ".xls", ".json"]
    DEFAULT_SAMPLE_SIZE: int = 5000  # For large visualization sampling
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

settings = Settings()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.DEMO_DIR, exist_ok=True)
