from typing import Optional
from aiogram.types import Voice, Message


def validate_voice_message(voice: Voice, max_duration: int = 60) -> tuple[bool, Optional[str]]:
    """
    Валидирует голосовое сообщение

    Args:
        voice: Голосовое сообщение
        max_duration: Максимальная длительность в секундах

    Returns:
        tuple: (is_valid, error_message)
    """
    if not voice:
        return False, "Голосовое сообщение не найдено"

    if voice.duration > max_duration:
        return False, f"Голосовое сообщение слишком длинное. Максимальная длительность: {max_duration} секунд"

    if voice.duration < 1:
        return False, "Голосовое сообщение слишком короткое"

    return True, None


def validate_text_input(text: str, min_length: int = 5, max_length: int = 1000) -> tuple[bool, Optional[str]]:
    """
    Валидирует текстовый ввод

    Args:
        text: Текст для валидации
        min_length: Минимальная длина
        max_length: Максимальная длина

    Returns:
        tuple: (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Текст не может быть пустым"

    text = text.strip()

    if len(text) < min_length:
        return False, f"Текст слишком короткий. Минимальная длина: {min_length} символов"

    if len(text) > max_length:
        return False, f"Текст слишком длинный. Максимальная длина: {max_length} символов"

    return True, None


def sanitize_text(text: str) -> str:
    """
    Очищает текст от лишних символов

    Args:
        text: Исходный текст

    Returns:
        str: Очищенный текст
    """
    if not text:
        return ""

    # Убираем лишние пробелы
    text = " ".join(text.split())

    # Убираем специальные символы, которые могут вызвать проблемы
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    return text.strip()
