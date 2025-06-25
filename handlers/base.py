from aiogram import Router, F
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, FSInputFile
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup


class ViolationStates(StatesGroup):
    """Состояния для обработки нарушений"""
    waiting_for_voice = State()


class BaseHandler:
    """Базовый обработчик с основной логикой бота"""

    def __init__(self):
        self.router = Router()
        self._setup_handlers()

    def _setup_handlers(self):
        """Настраивает обработчики"""
        self.router.message.register(self.start_command, Command("start"))
        self.router.message.register(self.help_command, Command("help"))
        self.router.callback_query.register(
            self.new_violation_callback, F.data == "new_violation")

    async def start_command(self, message: Message, state: FSMContext):
        """Обработчик команды /start"""
        welcome_text = """
🔍 **Привет, инспектор!**

Я помогу проанализировать нарушения и найти соответствующие нормативные документы.

💡 **Как использовать:**
1. Нажмите кнопку "Сообщить о нарушении"
2. Отправьте голосовое сообщение с описанием нарушения
3. Получите анализ с корректировкой формулировки и ссылками на нормативные документы

❗ **Поддерживаемые форматы:**
• Голосовые сообщения (до 60 секунд)
• Текстовые сообщения

Нажмите кнопку ниже, чтобы начать:
        """
        image_path = "assets/welcome_image.jpg"
        photo = FSInputFile(image_path)
        keyboard = self._create_main_keyboard()
        await message.answer_photo(
            photo=photo,
            caption=welcome_text,
            reply_markup=keyboard,
            parse_mode="Markdown"
        )
        await state.set_state(ViolationStates.waiting_for_voice)

    async def help_command(self, message: Message):
        """Обработчик команды /help"""
        help_text = """
📋 **Справка по использованию бота**

**Основные команды:**
/start - Запуск бота
/help - Показать эту справку

**Процесс работы:**
1. **Отправка нарушения** - Нажмите "Сообщить о нарушении" и отправьте голосовое сообщение
2. **Обработка** - Бот преобразует речь в текст и проанализирует нарушение
3. **Результат** - Вы получите:
   • Скорректированное описание нарушения
   • Ссылку на нормативный документ
   • Предложения по устранению

**Требования к голосовым сообщениям:**
• Длительность: до 60 секунд
• Язык: русский
• Качество: четкая речь

**Примеры нарушений:**
• "Отсутствие огнетушителя в помещении"
• "Неисправная электропроводка"
• "Отсутствие знаков безопасности"

Если возникли проблемы, попробуйте отправить сообщение заново.
        """

        keyboard = self._create_main_keyboard()
        await message.answer(help_text, reply_markup=keyboard, parse_mode="Markdown")

    async def new_violation_callback(self, callback_query, state: FSMContext):
        """Обработчик нажатия кнопки 'Сообщить о нарушении'"""
        await callback_query.answer()

        instruction_text = """
🎤 **Отправьте голосовое сообщение**

Опишите нарушение голосовым сообщением. 

❗**Рекомендации:**
• Говорите четко и понятно
• Опишите конкретное нарушение
• Укажите место и обстоятельства

**Примеры:**
• "В цехе отсутствует огнетушитель"
• "На лестнице нет перил"
• "Электропроводка не изолирована"

⏱️ **Максимальная длительность:** 60 секунд
        """

        await state.set_state(ViolationStates.waiting_for_voice)
        await callback_query.message.answer(instruction_text, parse_mode="Markdown")

    def _create_main_keyboard(self) -> InlineKeyboardMarkup:
        """Создает основную клавиатуру с кнопкой 'Сообщить о нарушении'"""
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
        """Возвращает роутер для регистрации в диспетчере"""
        return self.router
