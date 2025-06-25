import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Загружаем переменные окружения из .env файла
load_dotenv()


class Settings(BaseSettings):
    """Настройки приложения"""

    # Telegram Bot Configuration
    telegram_bot_token: str

    # OpenAI API Configuration
    openai_api_key: str
    openai_model: str = "gpt-4o"

    # Google Sheets Configuration
    google_sheets_credentials_file: str = "affable-elf-453008-v9-96d823bfa264.json"
    google_sheets_spreadsheet_id: str = "1ic6JMIqAJbOU3gzceuuSw2HbxTjsrCDSPeGLo3N7xiY"

    # Database Configuration
    rag_database_path: str = "chroma_db"

    # Application Configuration
    debug: bool = False
    log_level: str = "INFO"
    max_voice_duration: int = 60
    max_retries: int = 3

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }


# Создаем глобальный экземпляр настроек
settings = Settings()


def validate_settings():
    """Проверяет корректность настроек"""
    required_fields = [
        "telegram_bot_token",
        "openai_api_key"
    ]

    missing_fields = []
    for field in required_fields:
        if not getattr(settings, field):
            missing_fields.append(field)

    if missing_fields:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_fields)}")

    return True
