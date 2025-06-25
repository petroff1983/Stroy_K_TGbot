import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from services.logger import LoggerService
from models.violation import Violation, RAGChunk, ViolationResponse


class TestLoggerService:
    """Тесты для LoggerService"""

    @pytest.fixture
    def logger_service(self):
        """Фикстура для создания экземпляра LoggerService"""
        return LoggerService()

    @pytest.fixture
    def mock_violation(self):
        """Фикстура для создания тестового нарушения"""
        return Violation(
            original_text="отсутствие огнетушителя",
            corrected_description="Нарушение требований пожарной безопасности",
            document_title="СП 9.13130.2009",
            document_number="9.13130.2009",
            clause_number="4.1.3",
            suggestions="Установить огнетушитель",
            chunks=[
                {
                    "document_id": "1",
                    "document_title": "СП 9.13130.2009",
                    "document_number": "9.13130.2009",
                    "clause_number": "4.1.3",
                    "clause_text": "Огнетушители должны размещаться",
                    "relevance_score": 0.95
                }
            ],
            timestamp=datetime(2024, 1, 1, 12, 0, 0)
        )

    @pytest.fixture
    def mock_ai_response(self):
        """Фикстура для создания тестового AI ответа"""
        return ViolationResponse(
            corrected_description="Нарушение требований пожарной безопасности",
            document_info="СП 9.13130.2009, 9.13130.2009, 4.1.3",
            suggestions="Установить огнетушитель типа ОП-4",
            success=True
        )

    @pytest.mark.asyncio
    async def test_log_violation_success(self, logger_service, mock_violation, mock_ai_response):
        """Тест успешного логирования нарушения"""
        # Мокаем клиент Google Sheets
        mock_client = AsyncMock()
        mock_spreadsheet = AsyncMock()
        mock_worksheet = AsyncMock()

        mock_client.open_by_key.return_value = mock_spreadsheet
        mock_spreadsheet.get_worksheet.return_value = mock_worksheet
        mock_worksheet.append_row = AsyncMock()

        response_text = "Полный ответ ассистента"
        model_name = "gpt-4o-mini"

        with patch.object(logger_service, '_get_client', return_value=mock_client):
            result = await logger_service.log_violation(mock_violation, response_text, model_name)

        assert result is True
        mock_worksheet.append_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_violation_exception(self, logger_service, mock_violation, mock_ai_response):
        """Тест обработки исключения при логировании"""
        response_text = "Полный ответ ассистента"
        model_name = "gpt-4o-mini"
        with patch.object(logger_service, '_get_client', side_effect=Exception("Sheets Error")):
            result = await logger_service.log_violation(mock_violation, response_text, model_name)
        assert result is False

    @pytest.mark.asyncio
    async def test_log_error_success(self, logger_service):
        """Тест успешного логирования ошибки"""
        mock_client = AsyncMock()
        mock_spreadsheet = AsyncMock()
        mock_worksheet = AsyncMock()

        mock_client.open_by_key.return_value = mock_spreadsheet
        mock_spreadsheet.get_worksheet.return_value = mock_worksheet
        mock_worksheet.append_row = AsyncMock()

        with patch.object(logger_service, '_get_client', return_value=mock_client):
            result = await logger_service.log_error("Тестовая ошибка", 12345)

        assert result is True
        mock_worksheet.append_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_error_exception(self, logger_service):
        """Тест обработки исключения при логировании ошибки"""
        with patch.object(logger_service, '_get_client', side_effect=Exception("Sheets Error")):
            result = await logger_service.log_error("Тестовая ошибка")

        assert result is False

    @pytest.mark.asyncio
    async def test_test_connection_success(self, logger_service):
        """Тест успешного подключения к Google Sheets"""
        mock_client = AsyncMock()
        mock_spreadsheet = AsyncMock()
        mock_worksheet = AsyncMock()

        mock_client.open_by_key.return_value = mock_spreadsheet
        mock_spreadsheet.get_worksheet.return_value = mock_worksheet
        mock_worksheet.row_values = AsyncMock(
            return_value=["Timestamp", "Original_Text", "Corrected_Description"])

        with patch.object(logger_service, '_get_client', return_value=mock_client):
            result = await logger_service.test_connection()

        assert result is True

    @pytest.mark.asyncio
    async def test_test_connection_failure(self, logger_service):
        """Тест неудачного подключения к Google Sheets"""
        with patch.object(logger_service, '_get_client', side_effect=Exception("Connection Error")):
            result = await logger_service.test_connection()

        assert result is False

    def test_format_violation_data_success(self, logger_service, mock_violation, mock_ai_response):
        """Тест форматирования данных нарушения"""
        response_text = "Полный ответ ассистента"
        model_name = "gpt-4o-mini"
        row_data = logger_service._format_violation_data(
            mock_violation, response_text, model_name)

        assert len(row_data) == 6  # Количество столбцов
        assert row_data[0] == "2024-01-01 12:00:00"  # Timestamp
        assert row_data[1] == "отсутствие огнетушителя"  # Original_Text
        assert row_data[2] == response_text  # Полный ответ ассистента
        assert model_name in row_data[4]  # Модель GPT
        assert "СП 9.13130.2009" in row_data[3]  # Чанки в одном столбце
        assert row_data[5] == ""  # Error (пустой)

    def test_format_violation_data_with_error(self, logger_service, mock_violation):
        """Тест форматирования данных нарушения с ошибкой"""
        mock_violation.error = "Тестовая ошибка"
        response_text = "Полный ответ ассистента"
        model_name = "gpt-4o-mini"
        row_data = logger_service._format_violation_data(
            mock_violation, response_text, model_name)
        assert row_data[5] == "Тестовая ошибка"  # Error
        assert row_data[2] == response_text  # Полный ответ ассистента
        assert row_data[4] == model_name  # Модель GPT
        assert row_data[3]  # Чанки (могут быть пустыми)

    def test_format_chunks_success(self, logger_service):
        """Тест форматирования чанков"""
        chunks = [
            {
                "document_id": "1",
                "document_title": "СП 9.13130.2009",
                "document_number": "9.13130.2009",
                "clause_number": "4.1.3",
                "clause_text": "Огнетушители должны размещаться в легкодоступных местах",
                "relevance_score": 0.95
            },
            {
                "document_id": "2",
                "document_title": "СП 9.13130.2009",
                "document_number": "9.13130.2009",
                "clause_number": "4.1.4",
                "clause_text": "Расстояние от возможного очага пожара до огнетушителя",
                "relevance_score": 0.85
            }
        ]

        formatted_chunks = logger_service._format_chunks(chunks)

        assert len(formatted_chunks) == 3  # Всегда 3 элемента
        assert "СП 9.13130.2009" in formatted_chunks[0]
        assert "4.1.3" in formatted_chunks[0]
        assert "СП 9.13130.2009" in formatted_chunks[1]
        assert "4.1.4" in formatted_chunks[1]
        assert formatted_chunks[2] == ""  # Пустой третий чанк
        # Проверяем объединение для таблицы
        joined = '\n---\n'.join([c for c in formatted_chunks if c])
        assert "СП 9.13130.2009" in joined

    def test_format_chunks_empty(self, logger_service):
        """Тест форматирования пустых чанков"""
        formatted_chunks = logger_service._format_chunks([])

        assert len(formatted_chunks) == 3
        assert all(chunk == "" for chunk in formatted_chunks)

    def test_format_chunks_more_than_three(self, logger_service):
        """Тест форматирования более трех чанков"""
        chunks = [
            {"document_id": "1", "document_title": "Док1", "document_number": "1",
                "clause_number": "1", "clause_text": "Текст1", "relevance_score": 0.9},
            {"document_id": "2", "document_title": "Док2", "document_number": "2",
                "clause_number": "2", "clause_text": "Текст2", "relevance_score": 0.8},
            {"document_id": "3", "document_title": "Док3", "document_number": "3",
                "clause_number": "3", "clause_text": "Текст3", "relevance_score": 0.7},
            {"document_id": "4", "document_title": "Док4", "document_number": "4",
                "clause_number": "4", "clause_text": "Текст4", "relevance_score": 0.6}
        ]

        formatted_chunks = logger_service._format_chunks(chunks)

        assert len(formatted_chunks) == 3
        assert "Док1" in formatted_chunks[0]
        assert "Док2" in formatted_chunks[1]
        assert "Док3" in formatted_chunks[2]
        # Четвертый чанк должен быть отброшен

    @pytest.mark.asyncio
    async def test_get_client_singleton(self, logger_service):
        """Тест синглтона клиента"""
        with patch('services.logger.Credentials.from_service_account_file') as mock_creds:
            with patch('services.logger.gspread_asyncio.AsyncioGspreadClientManager') as mock_manager:
                mock_client_manager = AsyncMock()
                mock_client_manager.authorize = AsyncMock(
                    return_value="authorized_client")
                mock_manager.return_value = mock_client_manager

                client1 = await logger_service._get_client()
                client2 = await logger_service._get_client()

                assert client1 == client2
                assert client1 == "authorized_client"
                # authorize должен вызываться каждый раз
                assert mock_client_manager.authorize.call_count == 2
