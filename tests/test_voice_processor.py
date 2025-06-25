import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from aiogram.types import Voice, File
from aiogram import Bot
from services.voice_processor import VoiceProcessor


class TestVoiceProcessor:
    """Тесты для VoiceProcessor"""

    @pytest.fixture
    def voice_processor(self):
        """Фикстура для создания экземпляра VoiceProcessor"""
        with patch('services.voice_processor.openai.AsyncOpenAI'):
            return VoiceProcessor()

    @pytest.fixture
    def mock_voice(self):
        """Фикстура для создания мок-голосового сообщения"""
        voice = Mock(spec=Voice)
        voice.duration = 30
        voice.file_id = "test_file_id"
        return voice

    @pytest.fixture
    def mock_bot(self):
        """Фикстура для создания мок-бота"""
        bot = AsyncMock(spec=Bot)
        mock_file = Mock(spec=File)
        mock_file.file_path = "test/path/file.ogg"
        bot.get_file = AsyncMock(return_value=mock_file)
        bot.download_file = AsyncMock()
        return bot

    @pytest.mark.asyncio
    async def test_process_voice_message_success(self, voice_processor, mock_voice, mock_bot):
        """Тест успешной обработки голосового сообщения"""
        # Мокаем скачивание файла
        with patch('tempfile.NamedTemporaryFile', new_callable=MagicMock) as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.ogg"

            # Мокаем OpenAI ответ
            mock_response = Mock()
            mock_response.text = "Отсутствие огнетушителя в помещении"
            voice_processor.client.audio.transcriptions.create = AsyncMock(
                return_value=mock_response)

            # Мокаем открытие и удаление файла
            with patch('builtins.open', new_callable=MagicMock), patch('os.unlink'):
                success, text, error = await voice_processor.process_voice_message(mock_voice, mock_bot)

        assert success is True
        assert text == "Отсутствие огнетушителя в помещении"
        assert error is None

    @pytest.mark.asyncio
    async def test_process_voice_message_download_failure(self, voice_processor, mock_voice, mock_bot):
        """Тест ошибки скачивания голосового сообщения"""
        mock_bot.get_file = AsyncMock(side_effect=Exception("Download failed"))

        success, text, error = await voice_processor.process_voice_message(mock_voice, mock_bot)

        assert success is False
        assert text is None
        assert "скачать" in error.lower()

    @pytest.mark.asyncio
    async def test_process_voice_message_transcription_failure(self, voice_processor, mock_voice, mock_bot):
        """Тест ошибки распознавания речи"""
        with patch('tempfile.NamedTemporaryFile', new_callable=MagicMock) as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.ogg"

            # Мокаем ошибку OpenAI
            voice_processor.client.audio.transcriptions.create = AsyncMock(
                side_effect=Exception("API Error"))

            with patch('builtins.open', new_callable=MagicMock), patch('os.unlink'):
                success, text, error = await voice_processor.process_voice_message(mock_voice, mock_bot)

        assert success is False
        assert text is None
        assert "распознать" in error.lower()

    @pytest.mark.asyncio
    async def test_process_voice_message_empty_text(self, voice_processor, mock_voice, mock_bot):
        """Тест обработки пустого текста"""
        with patch('tempfile.NamedTemporaryFile', new_callable=MagicMock) as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.ogg"

            # Мокаем пустой ответ
            mock_response = Mock()
            mock_response.text = ""
            voice_processor.client.audio.transcriptions.create = AsyncMock(
                return_value=mock_response)

            with patch('builtins.open', new_callable=MagicMock), patch('os.unlink'):
                success, text, error = await voice_processor.process_voice_message(mock_voice, mock_bot)

        assert success is False
        assert text is None
        assert "распознать" in error.lower()

    @pytest.mark.asyncio
    async def test_download_voice_success(self, voice_processor, mock_voice, mock_bot):
        """Тест успешного скачивания голосового сообщения"""
        with patch('tempfile.NamedTemporaryFile', new_callable=MagicMock) as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.ogg"

            result = await voice_processor._download_voice(mock_voice, mock_bot)

            assert result == "/tmp/test.ogg"
            mock_bot.download_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_voice_failure(self, voice_processor, mock_voice, mock_bot):
        """Тест ошибки скачивания"""
        mock_bot.get_file = AsyncMock(side_effect=Exception("Download error"))

        result = await voice_processor._download_voice(mock_voice, mock_bot)

        assert result is None

    @pytest.mark.asyncio
    async def test_transcribe_audio_success(self, voice_processor):
        """Тест успешного преобразования аудио в текст"""
        mock_response = Mock()
        mock_response.text = "Тестовый текст"
        voice_processor.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response)

        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value = Mock()

            result = await voice_processor._transcribe_audio("/tmp/test.ogg")

            assert result == "Тестовый текст"

    @pytest.mark.asyncio
    async def test_transcribe_audio_failure(self, voice_processor):
        """Тест ошибки преобразования аудио"""
        voice_processor.client.audio.transcriptions.create = AsyncMock(
            side_effect=Exception("API Error"))

        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value = Mock()

            result = await voice_processor._transcribe_audio("/tmp/test.ogg")

            assert result is None
