import asyncio
from aiogram import Router, F
from aiogram.types import Message, Voice, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

from services.voice_processor import VoiceProcessor
from services.rag_service import RAGService
from services.ai_service import AIService
from services.logger import LoggerService
from models.violation import Violation
from utils.validators import validate_voice_message, validate_text_input
from handlers.base import ViolationStates


class VoiceHandler:
    """Обработчик голосовых сообщений"""

    def __init__(self):
        self.router = Router()
        self.voice_processor = VoiceProcessor()
        self.rag_service = RAGService()
        self.ai_service = AIService()
        self.logger_service = LoggerService()
        self._setup_handlers()

    def _setup_handlers(self):
        """Настраивает обработчики"""
        self.router.message.register(
            self.handle_voice_message,
            F.voice,
            ViolationStates.waiting_for_voice
        )
        self.router.message.register(
            self.handle_text_message,
            F.text,
            ViolationStates.waiting_for_voice
        )
        self.router.message.register(
            self.handle_other_message,
            ViolationStates.waiting_for_voice
        )

    async def handle_voice_message(self, message: Message, state: FSMContext):
        """Обработчик голосовых сообщений"""
        voice = message.voice

        # Валидируем голосовое сообщение
        is_valid, error_message = validate_voice_message(voice)
        if not is_valid:
            await message.answer(
                f"❌ {error_message}\n\nПопробуйте отправить голосовое сообщение заново.",
                reply_markup=self._create_main_keyboard()
            )
            await state.clear()
            return

        # Отправляем сообщение о начале обработки
        processing_msg = await message.answer("🔄 Обрабатываю голосовое сообщение...")

        try:
            # Обрабатываем голосовое сообщение
            success, text, error = await self.voice_processor.process_voice_message(voice, message.bot)

            if not success:
                await processing_msg.delete()
                await message.answer(
                    f"❌ Ошибка распознавания речи: {error}\n\nПопробуйте отправить сообщение заново или используйте текстовый ввод.",
                    reply_markup=self._create_main_keyboard()
                )
                await state.clear()
                return

            # Проверяем качество распознанного текста
            is_text_valid, text_error = validate_text_input(text)
            if not is_text_valid:
                await processing_msg.delete()
                await message.answer(
                    f"❌ Распознанный текст слишком короткий или неполный: {text_error}\n\nПопробуйте отправить более четкое голосовое сообщение.",
                    reply_markup=self._create_main_keyboard()
                )
                await state.clear()
                return

            # Обновляем сообщение о прогрессе
            await processing_msg.edit_text("🔍 Анализирую нарушение...")

            # Ищем релевантные документы в RAG базе
            rag_chunks = await self.rag_service.search_relevant_chunks(text)
            print(f"[VoiceHandler] rag_chunks: {rag_chunks}")

            # Анализируем нарушение с помощью AI
            ai_response = await self.ai_service.analyze_violation(text, rag_chunks)

            # Создаем объект нарушения для логирования
            doc_title, doc_num, clause_num = None, None, None
            if ai_response.success and ai_response.document_info:
                # Безопасно парсим информацию о документе
                parts = [p.strip()
                         for p in ai_response.document_info.split(',', 2)]
                doc_title = parts[0]
                doc_num = parts[1] if len(parts) > 1 else None
                clause_num = parts[2] if len(parts) > 2 else None

            violation = Violation(
                original_text=text,
                corrected_description=ai_response.corrected_description,
                document_title=doc_title,
                document_number=doc_num,
                clause_number=clause_num,
                suggestions=ai_response.suggestions,
                chunks=[chunk.dict() for chunk in rag_chunks],
                error=ai_response.error_message if not ai_response.success else None
            )

            # Формируем ответ пользователю
            response_text = self._format_response(ai_response, text)

            print('RAG chunks:', rag_chunks)
            print('violation.chunks:', violation.chunks)
            # Логируем в Google Sheets
            print("Перед логированием")
            await self.logger_service.log_violation(
                violation,
                response_text,
                self.ai_service.model  # model_name
            )
            print("После логирования")

            # Отправляем результат
            try:
                await processing_msg.delete()
            except:
                pass
            await message.answer(
                response_text,
                reply_markup=self._create_main_keyboard(),
                parse_mode="Markdown"
            )
            print("Ответ отправлен")

            # Очищаем состояние
            await state.clear()

        except Exception as e:
            try:
                await processing_msg.delete()
            except:
                pass
            await message.answer(
                f"❌ Произошла ошибка при обработке: {str(e)}\n\nПопробуйте позже.",
                reply_markup=self._create_main_keyboard()
            )
            await state.clear()

            # Логируем ошибку
            await self.logger_service.log_error(str(e), message.from_user.id)

    async def handle_text_message(self, message: Message, state: FSMContext):
        """Обработчик текстовых сообщений (для уточнений)"""
        text = message.text.strip()

        # Валидируем текст
        is_valid, error_message = validate_text_input(text)
        if not is_valid:
            await message.answer(
                f"❌ {error_message}\n\nПожалуйста, отправьте голосовое сообщение или более подробное текстовое описание.",
                reply_markup=self._create_main_keyboard()
            )
            await state.clear()
            return

        # Отправляем сообщение о начале обработки
        processing_msg = await message.answer("🔄 Анализирую нарушение...")

        try:
            # Ищем релевантные документы
            rag_chunks = await self.rag_service.search_relevant_chunks(text)
            print(f"[VoiceHandler] rag_chunks: {rag_chunks}")

            # Анализируем нарушение
            ai_response = await self.ai_service.analyze_violation(text, rag_chunks)

            # Создаем объект нарушения
            doc_title, doc_num, clause_num = None, None, None
            if ai_response.success and ai_response.document_info:
                # Безопасно парсим информацию о документе
                parts = [p.strip()
                         for p in ai_response.document_info.split(',', 2)]
                doc_title = parts[0]
                doc_num = parts[1] if len(parts) > 1 else None
                clause_num = parts[2] if len(parts) > 2 else None

            violation = Violation(
                original_text=text,
                corrected_description=ai_response.corrected_description,
                document_title=doc_title,
                document_number=doc_num,
                clause_number=clause_num,
                suggestions=ai_response.suggestions,
                chunks=[chunk.dict() for chunk in rag_chunks],
                error=ai_response.error_message if not ai_response.success else None
            )

            # Формируем ответ
            response_text = self._format_response(ai_response, text)

            print('RAG chunks:', rag_chunks)
            print('violation.chunks:', violation.chunks)
            # Логируем
            print("Перед логированием")
            await self.logger_service.log_violation(
                violation,
                response_text,
                self.ai_service.model  # model_name
            )
            print("После логирования")

            # Отправляем результат
            try:
                await processing_msg.delete()
            except:
                pass
            await message.answer(
                response_text,
                reply_markup=self._create_main_keyboard(),
                parse_mode="Markdown"
            )
            print("Ответ отправлен")

            await state.clear()

        except Exception as e:
            try:
                await processing_msg.delete()
            except:
                pass
            await message.answer(
                f"❌ Произошла ошибка при обработке: {str(e)}\n\nПопробуйте позже.",
                reply_markup=self._create_main_keyboard()
            )
            await state.clear()
            await self.logger_service.log_error(str(e), message.from_user.id)

    async def handle_other_message(self, message: Message, state: FSMContext):
        """Обработчик других типов сообщений"""
        await message.answer(
            "❌ Пожалуйста, отправьте голосовое сообщение или текстовое описание нарушения.",
            reply_markup=self._create_main_keyboard()
        )
        await state.clear()

    def _format_response(self, ai_response, original_text: str) -> str:
        """Форматирует ответ для пользователя"""
        if not ai_response.success:
            return f"""❌ **Ошибка анализа нарушения**

**Исходный текст:** {original_text}

**Ошибка:** {ai_response.error_message}

Попробуйте отправить более подробное описание нарушения."""

        return f"""✅ **Анализ нарушения завершен**

📝 **Исходный текст:**
`{original_text}`

✍️ **Скорректированное описание:**
{ai_response.corrected_description}

📖 **Нормативный документ:**
_{ai_response.document_info}_

❗ **Предлагаемые меры по устранению:**
{ai_response.suggestions}

---
Нажмите кнопку ниже для нового нарушения:"""

    def _create_main_keyboard(self) -> InlineKeyboardMarkup:
        """Создает основную клавиатуру"""
        from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="🚨 Сообщить о нарушении",
                        callback_data="new_violation"
                    )
                ]
            ]
        )
        return keyboard

    def get_router(self):
        """Возвращает роутер"""
        return self.router
