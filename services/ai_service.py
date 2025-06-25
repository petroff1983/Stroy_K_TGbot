import asyncio
import re
from typing import List, Optional
import openai
from models.violation import ViolationResponse, RAGChunk
from config import settings


class AIService:
    """Сервис для работы с OpenAI API"""

    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    async def analyze_violation(self, original_text: str, rag_chunks: List[RAGChunk]) -> ViolationResponse:
        """
        Анализирует нарушение и генерирует ответ

        Args:
            original_text: Исходный текст нарушения
            rag_chunks: Релевантные чанки из RAG базы

        Returns:
            ViolationResponse: Ответ с анализом нарушения
        """
        try:
            # Формируем контекст из RAG чанков
            context = self._format_rag_context(rag_chunks)

            # Генерируем промпт для анализа
            prompt = self._create_analysis_prompt(original_text, context)

            # Получаем ответ от OpenAI
            response = await self._get_openai_response(prompt)

            # Парсим ответ
            return self._parse_ai_response(response, rag_chunks)

        except Exception as e:
            return ViolationResponse(
                corrected_description="Ошибка анализа нарушения",
                document_info="Не удалось определить нормативный документ",
                suggestions="Попробуйте повторить описание нарушения",
                success=False,
                error_message=str(e)
            )

    def _format_rag_context(self, chunks: List[RAGChunk]) -> str:
        """
        Форматирует контекст из RAG чанков

        Args:
            chunks: Список чанков

        Returns:
            str: Отформатированный контекст
        """
        if not chunks:
            return "Релевантные нормативные документы не найдены."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"""
Документ {i}:
- Название: {chunk.document_title}
- Номер: {chunk.document_number}
- Пункт: {chunk.clause_number}
- Текст: {chunk.clause_text}
- Релевантность: {chunk.relevance_score:.3f}
""")

        return "\n".join(context_parts)

    def _create_analysis_prompt(self, original_text: str, context: str) -> str:
        return f"""
Ты — профессиональный ИИ-ассистент по строительному контролю. Твоя задача:
1. Кратко и четко скорректировать описание нарушения.
2. Найти и указать нормативный документ (название, номер, пункт).
3. Предложить КОНКРЕТНЫЕ меры по устранению с указанием срока (даже если приходится предполагать по типовой ситуации).
4. Не отвечай на вопросы вне строительного контроля — если вопрос не по теме, напиши: 'Я могу отвечать только на вопросы по строительному контролю и нормативам.'

Формат ответа:
**Скорректированное описание:** ...
**Нормативный документ:** ...
**Предлагаемые меры по устранению:** ...

Исходный текст: {original_text}
Контекст из RAG базы:
{context}
"""

    async def _get_openai_response(self, prompt: str) -> str:
        """
        Получает ответ от OpenAI API

        Args:
            prompt: Промпт для модели

        Returns:
            str: Ответ модели
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "ВЫ — ПРОФЕССИОНАЛЬНЫЙ ИИ-АССИСТЕНТ В СФЕРЕ СТРОИТЕЛЬНОГО КОНТРОЛЯ. ВАША ЗАДАЧА — ПРЕОБРАЗОВЫВАТЬ ОПИСАНИЯ НАРУШЕНИЙ В СТАНДАРТИЗИРОВАННЫЕ ФОРМУЛИРОВКИ С УЧЁТОМ АКТУАЛЬНОЙ НОРМАТИВНОЙ ДОКУМЕНТАЦИИ, ПОДБИРАТЬ ССЫЛКИ НА НОРМЫ, ПРЕДЛАГАТЬ МЕРЫ ПО УСТРАНЕНИЮ И ФОРМИРОВАТЬ ГОТОВЫЕ ПРЕДПИСАНИЯ."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            result = response.choices[0].message.content.strip()
            print("[AIService] Полный ответ OpenAI:\n", result)
            return result
        except Exception as e:
            raise Exception(f"Ошибка получения ответа от OpenAI: {str(e)}")

    def _parse_ai_response(self, response: str, chunks: List[RAGChunk]) -> ViolationResponse:
        """
        Парсит ответ от AI модели с использованием регулярных выражений.

        Args:
            response: Ответ от модели
            chunks: RAG чанки

        Returns:
            ViolationResponse: Структурированный ответ
        """
        try:
            # Используем более простые регулярные выражения
            corrected_description_match = re.search(
                r'Скорректированное описание:\s*(.*?)(?=\s*Нормативный документ:|$)',
                response, re.DOTALL | re.IGNORECASE
            )
            document_info_match = re.search(
                r'Нормативный документ:\s*(.*?)(?=\s*Предлагаемые меры по устранению:|$)',
                response, re.DOTALL | re.IGNORECASE
            )
            suggestions_match = re.search(
                r'Предлагаемые меры по устранению:\s*(.*?)(?=\s*\*\*|$)',
                response, re.DOTALL | re.IGNORECASE
            )

            # Извлекаем текст без префиксов
            corrected_description = ""
            if corrected_description_match:
                corrected_description = corrected_description_match.group(
                    1).strip()
                # Убираем префикс если есть
                if corrected_description.lower().startswith("скорректированное описание:"):
                    corrected_description = corrected_description[len(
                        "скорректированное описание:"):].strip()

            document_info = ""
            if document_info_match:
                document_info = document_info_match.group(1).strip()
                # Убираем префикс если есть
                if document_info.lower().startswith("нормативный документ:"):
                    document_info = document_info[len(
                        "нормативный документ:"):].strip()

            suggestions = ""
            if suggestions_match:
                suggestions = suggestions_match.group(1).strip()
                # Убираем префикс если есть
                if suggestions.lower().startswith("предлагаемые меры по устранению:"):
                    suggestions = suggestions[len(
                        "предлагаемые меры по устранению:"):].strip()

            # Если парсинг не дал результатов, возможно, модель ответила не по формату
            if not corrected_description and not document_info and not suggestions:
                # В качестве запасного варианта берем весь ответ, чтобы не терять информацию
                corrected_description = response

            # Если документ не найден в ответе, но есть чанки из RAG, берем лучший
            if not document_info and chunks:
                best_chunk = chunks[0]
                document_info = f"{best_chunk.document_title}, {best_chunk.document_number}, {best_chunk.clause_number}"

            if not suggestions:
                suggestions = "Требуется дополнительный анализ для определения мер по устранению"

            return ViolationResponse(
                corrected_description=corrected_description,
                document_info=document_info,
                suggestions=suggestions,
                success=True
            )

        except Exception as e:
            return ViolationResponse(
                corrected_description="Ошибка обработки ответа",
                document_info="Не определено",
                suggestions="Требуется повторный анализ",
                success=False,
                error_message=str(e)
            )
