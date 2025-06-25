import asyncio
from datetime import datetime
from typing import List, Optional
import gspread_asyncio
from google.oauth2.service_account import Credentials
from models.violation import Violation, RAGChunk
from config import settings


class LoggerService:
    """Сервис для логирования в Google Sheets"""

    def __init__(self):
        self.credentials_file = settings.google_sheets_credentials_file
        self.spreadsheet_id = settings.google_sheets_spreadsheet_id
        self._client = None

    async def _get_client(self):
        """Получает клиент для работы с Google Sheets"""
        if self._client is None:
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]

            def get_creds():
                return Credentials.from_service_account_file(
                    self.credentials_file,
                    scopes=scopes
                )

            self._client = gspread_asyncio.AsyncioGspreadClientManager(
                get_creds)

        return await self._client.authorize()

    async def log_violation(self, violation: Violation, response_text: str, model_name: str) -> bool:
        """
        Логирует нарушение в Google Sheets

        Args:
            violation: Данные нарушения
            response_text: Полный текст ответа ассистента
            model_name: Название модели GPT

        Returns:
            bool: Успешность логирования
        """
        try:
            client = await self._get_client()
            spreadsheet = await client.open_by_key(self.spreadsheet_id)
            worksheet = await spreadsheet.get_worksheet(0)  # Первый лист

            # Формируем данные для записи
            row_data = self._format_violation_data(
                violation, response_text, model_name)

            # Добавляем строку в таблицу
            await worksheet.append_row(row_data)

            return True

        except Exception as e:
            print(f"Ошибка логирования в Google Sheets: {e}")
            return False

    def _format_violation_data(self, violation: Violation, response_text: str, model_name: str) -> list:
        """
        Форматирует данные нарушения для записи в таблицу

        Args:
            violation: Данные нарушения
            response_text: Полный текст ответа ассистента
            model_name: Название модели GPT

        Returns:
            list: Список значений для строки
        """
        # Форматируем чанки
        chunks_text = self._format_chunks(violation.chunks)
        joined_chunks = '\n\n====================\n\n'.join(
            [c for c in chunks_text if c])

        # Форматируем время
        timestamp = violation.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # Формируем строку данных
        row_data = [
            timestamp,  # Дата
            "",        # D пользователя (пусто, если нет user_id)
            violation.original_text,  # Вопрос
            response_text,  # Ответ ассистента
            joined_chunks,  # Текст чанка
            model_name      # Модель GPT
        ]

        return row_data

    def _format_chunks(self, chunks: list) -> list:
        """
        Форматирует чанки для записи в таблицу

        Args:
            chunks: Список чанков (словарей)

        Returns:
            list: Отформатированные чанки
        """
        formatted_chunks = []

        # Берем только первые 3 чанка
        for idx, chunk in enumerate(chunks[:3], 1):
            text = chunk.get('clause_text', '')
            if len(text) > 200:
                text = text[:200] + '...'
            chunk_text = (
                f"Чанк {idx}:\n"
                f"Документ: {chunk.get('document_title', '')} {chunk.get('document_number', '')}, п.{chunk.get('clause_number', '')}\n"
                f"Текст: {text}"
            )
            formatted_chunks.append(chunk_text)

        # Дополняем до 3 элементов пустыми строками
        while len(formatted_chunks) < 3:
            formatted_chunks.append("")

        return formatted_chunks

    async def log_error(self, error_message: str, user_id: Optional[int] = None) -> bool:
        """
        Логирует ошибку в Google Sheets

        Args:
            error_message: Сообщение об ошибке
            user_id: ID пользователя (опционально)

        Returns:
            bool: Успешность логирования
        """
        try:
            client = await self._get_client()
            spreadsheet = await client.open_by_key(self.spreadsheet_id)
            worksheet = await spreadsheet.get_worksheet(0)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            error_row = [
                timestamp,  # Время
                # Исходный текст
                f"Ошибка пользователя {user_id}" if user_id else "Системная ошибка",
                "",  # Скорректированное описание
                "",  # Нормативный документ
                "",  # Предлагаемые меры
                "",  # Чанк 1
                "",  # Чанк 2
                "",  # Чанк 3
                error_message  # Ошибка
            ]

            await worksheet.append_row(error_row)
            return True

        except Exception as e:
            print(f"Ошибка логирования ошибки в Google Sheets: {e}")
            return False

    async def test_connection(self) -> bool:
        """
        Тестирует подключение к Google Sheets

        Returns:
            bool: Успешность подключения
        """
        try:
            client = await self._get_client()
            spreadsheet = await client.open_by_key(self.spreadsheet_id)
            worksheet = await spreadsheet.get_worksheet(0)

            # Пытаемся получить заголовки
            headers = await worksheet.row_values(1)

            return True

        except Exception as e:
            print(f"Ошибка подключения к Google Sheets: {e}")
            return False
