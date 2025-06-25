import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

from config import settings, validate_settings
from handlers.base import BaseHandler
from handlers.voice import VoiceHandler
from services.logger import LoggerService

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Главная функция приложения"""
    try:
        # Проверяем настройки
        validate_settings()
        logger.info("Настройки валидны")

        # Инициализируем бота и диспетчер
        bot = Bot(
            token=settings.telegram_bot_token,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML)
        )
        storage = MemoryStorage()
        dp = Dispatcher(storage=storage)

        # Инициализируем обработчики
        base_handler = BaseHandler()
        voice_handler = VoiceHandler()

        # Регистрируем роутеры
        dp.include_router(base_handler.get_router())
        dp.include_router(voice_handler.get_router())

        # Тестируем подключение к Google Sheets
        logger_service = LoggerService()
        sheets_connected = await logger_service.test_connection()
        if sheets_connected:
            logger.info("Подключение к Google Sheets успешно")
        else:
            logger.warning("Не удалось подключиться к Google Sheets")

        logger.info("Бот запущен")

        # Запускаем бота
        await dp.start_polling(bot)

    except Exception as e:
        logger.error(f"Ошибка запуска бота: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
