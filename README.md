Привет, это тестовое изменение.

# Stroy_K - Telegram Bot для анализа нарушений

Telegram-бот для анализа нарушений в строительной сфере с использованием AI и RAG-технологий.

## 🚀 Возможности

- **Голосовые сообщения**: Принимает голосовые сообщения и преобразует их в текст с помощью Whisper
- **AI-анализ**: Анализирует нарушения с использованием OpenAI GPT
- **RAG-база данных**: Использует ChromaDB для поиска релевантной нормативной документации
- **Логирование**: Автоматически логирует все данные в Google Sheets
- **Русский язык**: Полная поддержка русского языка

## 📋 Требования

- Python 3.8+
- Telegram Bot Token
- OpenAI API Key
- Google Sheets API credentials

## 🛠 Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd Stroy_K
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Создайте файл `.env` с вашими ключами:
```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key
GOOGLE_SHEETS_CREDENTIALS_FILE=affable-elf-453008-v9-96d823bfa264.json
GOOGLE_SHEETS_SPREADSHEET_ID=your_spreadsheet_id
RAG_DATABASE_PATH=chroma_db
```

5. Запустите бота:
```bash
python main.py
```

## 🏗 Архитектура

```
Stroy_K/
├── main.py                 # Главный файл бота
├── config.py              # Конфигурация
├── requirements.txt       # Зависимости
├── handlers/             # Обработчики сообщений
│   ├── base.py          # Базовый обработчик
│   └── voice.py         # Обработчик голосовых сообщений
├── services/            # Сервисы
│   ├── ai_service.py    # AI-сервис (OpenAI)
│   ├── rag_service.py   # RAG-сервис (ChromaDB)
│   ├── logger.py        # Логирование в Google Sheets
│   └── voice_processor.py # Обработка голоса
├── models/              # Модели данных
│   └── violation.py     # Модель нарушения
├── utils/               # Утилиты
│   └── validators.py    # Валидаторы
└── tests/               # Тесты
    ├── test_ai_service.py
    ├── test_rag_service.py
    ├── test_logger.py
    └── test_voice_processor.py
```

## 🧪 Тестирование

Запустите тесты:
```bash
pytest
```

## 📝 Использование

1. Найдите бота в Telegram
2. Нажмите кнопку "Сообщить о нарушении"
3. Отправьте голосовое или текстовое сообщение с описанием нарушения
4. Получите анализ с рекомендациями по нормативной документации

## 🔧 Конфигурация

Основные настройки в `config.py`:
- Пути к базам данных
- Настройки API
- Параметры RAG-поиска

## 📊 Логирование

Все обращения к боту автоматически записываются в Google Sheets с полной информацией:
- Время обращения
- Текст сообщения
- Релевантные документы
- Анализ AI
- Рекомендации

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Внесите изменения
4. Создайте Pull Request

## 📄 Лицензия

MIT License

## 📞 Поддержка

По вопросам работы бота обращайтесь к разработчику. 