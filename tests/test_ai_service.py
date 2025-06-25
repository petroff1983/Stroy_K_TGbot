import pytest
from unittest.mock import Mock, AsyncMock, patch
from services.ai_service import AIService
from models.violation import ViolationResponse, RAGChunk


class TestAIService:
    """Тесты для AIService"""

    @pytest.fixture
    def ai_service(self):
        """Фикстура для создания экземпляра AIService"""
        with patch('services.ai_service.openai.AsyncOpenAI'):
            return AIService()

    @pytest.fixture
    def mock_chunks(self):
        """Фикстура для создания тестовых чанков"""
        return [
            RAGChunk(
                document_id="1",
                document_title="СП 9.13130.2009",
                document_number="9.13130.2009",
                clause_number="4.1.3",
                clause_text="Огнетушители должны размещаться в легкодоступных местах",
                relevance_score=0.95
            ),
            RAGChunk(
                document_id="2",
                document_title="СП 9.13130.2009",
                document_number="9.13130.2009",
                clause_number="4.1.4",
                clause_text="Расстояние от возможного очага пожара до огнетушителя",
                relevance_score=0.85
            )
        ]

    @pytest.mark.asyncio
    async def test_analyze_violation_success(self, ai_service, mock_chunks):
        """Тест успешного анализа нарушения"""
        original_text = "отсутствие огнетушителя в помещении"

        # Мокаем ответ от OpenAI
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
Скорректированное описание: Нарушение требований пожарной безопасности, связанное с отсутствием огнетушителя в помещении
Нормативный документ: СП 9.13130.2009, 9.13130.2009, 4.1.3
Предлагаемые меры по устранению: Установить огнетушитель типа ОП-4 в помещении в соответствии с требованиями СП 9.13130.2009
"""

        ai_service.client.chat.completions.create = AsyncMock(
            return_value=mock_response)

        result = await ai_service.analyze_violation(original_text, mock_chunks)

        assert result.success is True
        assert "Нарушение требований пожарной безопасности" in result.corrected_description
        assert "СП 9.13130.2009" in result.document_info
        assert "Установить огнетушитель" in result.suggestions

    @pytest.mark.asyncio
    async def test_analyze_violation_exception(self, ai_service, mock_chunks):
        """Тест обработки исключения при анализе"""
        original_text = "отсутствие огнетушителя"

        ai_service.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error"))

        result = await ai_service.analyze_violation(original_text, mock_chunks)

        assert result.success is False
        assert "Ошибка анализа нарушения" in result.corrected_description
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_analyze_violation_empty_chunks(self, ai_service):
        """Тест анализа нарушения без релевантных чанков"""
        original_text = "неизвестное нарушение"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
Скорректированное описание: Неизвестное нарушение
Нормативный документ: Не определено
Предлагаемые меры по устранению: Требуется дополнительный анализ
"""

        ai_service.client.chat.completions.create = AsyncMock(
            return_value=mock_response)

        result = await ai_service.analyze_violation(original_text, [])

        assert result.success is True
        assert "Неизвестное нарушение" in result.corrected_description

    def test_format_rag_context_with_chunks(self, ai_service, mock_chunks):
        """Тест форматирования контекста с чанками"""
        context = ai_service._format_rag_context(mock_chunks)

        assert "Документ 1:" in context
        assert "СП 9.13130.2009" in context
        assert "4.1.3" in context
        assert "0.950" in context

    def test_format_rag_context_empty(self, ai_service):
        """Тест форматирования пустого контекста"""
        context = ai_service._format_rag_context([])

        assert "Релевантные нормативные документы не найдены" in context

    def test_create_analysis_prompt(self, ai_service, mock_chunks):
        """Тест создания промпта для анализа"""
        original_text = "отсутствие огнетушителя"
        context = ai_service._format_rag_context(mock_chunks)

        prompt = ai_service._create_analysis_prompt(original_text, context)

        assert original_text in prompt
        assert "Скорректированное описание:" in prompt
        assert "Нормативный документ:" in prompt
        assert "Предлагаемые меры по устранению:" in prompt

    @pytest.mark.asyncio
    async def test_get_openai_response_success(self, ai_service):
        """Тест успешного получения ответа от OpenAI"""
        prompt = "Тестовый промпт"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Тестовый ответ"

        ai_service.client.chat.completions.create = AsyncMock(
            return_value=mock_response)

        result = await ai_service._get_openai_response(prompt)

        assert result == "Тестовый ответ"
        ai_service.client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_openai_response_exception(self, ai_service):
        """Тест обработки исключения при получении ответа от OpenAI"""
        prompt = "Тестовый промпт"

        ai_service.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error"))

        with pytest.raises(Exception) as exc_info:
            await ai_service._get_openai_response(prompt)

        assert "Ошибка получения ответа от OpenAI" in str(exc_info.value)

    def test_parse_ai_response_success(self, ai_service, mock_chunks):
        """Тест успешного парсинга ответа AI"""
        response = """
Скорректированное описание: Нарушение требований пожарной безопасности
Нормативный документ: СП 9.13130.2009, 9.13130.2009, 4.1.3
Предлагаемые меры по устранению: Установить огнетушитель
"""

        result = ai_service._parse_ai_response(response, mock_chunks)

        assert result.success is True
        assert result.corrected_description == "Нарушение требований пожарной безопасности"
        assert result.document_info == "СП 9.13130.2009, 9.13130.2009, 4.1.3"
        assert result.suggestions == "Установить огнетушитель"

    def test_parse_ai_response_partial(self, ai_service, mock_chunks):
        """Тест парсинга частичного ответа AI"""
        response = "Скорректированное описание: Нарушение требований пожарной безопасности"

        result = ai_service._parse_ai_response(response, mock_chunks)

        assert result.success is True
        assert result.corrected_description == "Нарушение требований пожарной безопасности"
        # Должен использовать информацию из чанка
        assert "СП 9.13130.2009" in result.document_info

    def test_parse_ai_response_empty(self, ai_service, mock_chunks):
        """Тест парсинга пустого ответа AI"""
        response = ""

        result = ai_service._parse_ai_response(response, mock_chunks)

        assert result.success is True
        assert result.corrected_description == ""
        assert "СП 9.13130.2009" in result.document_info
        assert "Требуется дополнительный анализ" in result.suggestions

    def test_parse_ai_response_exception(self, ai_service, mock_chunks):
        """Тест обработки исключения при парсинге"""
        response = "Некорректный ответ"

        with patch('services.ai_service.AIService._parse_ai_response', side_effect=Exception("Parse error")):
            with pytest.raises(Exception) as exc_info:
                ai_service._parse_ai_response(response, mock_chunks)

            assert "Parse error" in str(exc_info.value)

    def test_parse_ai_response_no_chunks(self, ai_service):
        """Тест парсинга ответа без чанков"""
        response = """
Скорректированное описание: Нарушение требований
Нормативный документ: Не определено
Предлагаемые меры по устранению: Требуется анализ
"""

        result = ai_service._parse_ai_response(response, [])

        assert result.success is True
        assert result.corrected_description == "Нарушение требований"
        assert result.document_info == "Не определено"
        assert result.suggestions == "Требуется анализ"
