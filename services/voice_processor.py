import asyncio
import tempfile
import os
from typing import Optional
import openai
from aiogram.types import Voice
from aiogram import Bot
from config import settings


class VoiceProcessor:
    """Сервис для обработки голосовых сообщений"""

    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    async def process_voice_message(self, voice: Voice, bot: Bot) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Обрабатывает голосовое сообщение и преобразует его в текст

        Args:
            voice: Голосовое сообщение
            bot: Экземпляр бота

        Returns:
            tuple: (success, text, error_message)
        """
        try:
            # Скачиваем голосовое сообщение
            voice_file = await self._download_voice(voice, bot)
            if not voice_file:
                return False, None, "Не удалось скачать голосовое сообщение"

            # Преобразуем в текст с помощью Whisper
            text = await self._transcribe_audio(voice_file)

            # Удаляем временный файл
            os.unlink(voice_file)

            if not text or not text.strip():
                return False, None, "Не удалось распознать речь в голосовом сообщении"

            return True, text.strip(), None

        except Exception as e:
            return False, None, f"Ошибка обработки голосового сообщения: {str(e)}"

    async def _download_voice(self, voice: Voice, bot: Bot) -> Optional[str]:
        """
        Скачивает голосовое сообщение во временный файл

        Args:
            voice: Голосовое сообщение
            bot: Экземпляр бота

        Returns:
            str: Путь к временному файлу или None
        """
        try:
            # Создаем временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_file:
                # Получаем информацию о файле
                file_info = await bot.get_file(voice.file_id)
                # Скачиваем файл
                await bot.download_file(file_info.file_path, temp_file.name)
                return temp_file.name

        except Exception as e:
            print(f"Ошибка скачивания голосового сообщения: {e}")
            return None

    async def _transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """
        Преобразует аудио в текст с помощью Whisper

        Args:
            audio_file_path: Путь к аудио файлу

        Returns:
            str: Распознанный текст или None
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ru"
                )

                return response.text

        except Exception as e:
            return None
