import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from services.rag_service import RAGService
from models.violation import RAGChunk


class TestRAGService:
    """Тесты для RAGService"""

    @pytest.fixture
    def rag_service(self):
        """Создает экземпляр RAGService для тестов"""
        with patch('services.rag_service.chromadb.PersistentClient'):
            with patch('services.rag_service.SentenceTransformer'):
                service = RAGService()
                service.collection = Mock()
                return service

    @pytest.mark.asyncio
    async def test_search_relevant_chunks_success(self, rag_service):
        """Тест успешного поиска релевантных чанков"""
        # Мокаем результат ChromaDB
        mock_results = {
            'ids': [['doc1', 'doc2']],
            'metadatas': [[
                {
                    'document_title': 'СП 9.13130.2009',
                    'document_number': '9.13130.2009',
                    'clause_number': '4.1.3',
                    'keywords': 'огнетушитель'
                },
                {
                    'document_title': 'СП 9.13130.2009',
                    'document_number': '9.13130.2009',
                    'clause_number': '4.1.4',
                    'keywords': 'расстояние'
                }
            ]],
            'documents': [['Огнетушители должны размещаться', 'Расстояние от очага']],
            'distances': [[0.1, 0.2]]
        }

        rag_service.collection.query.return_value = mock_results

        result = await rag_service.search_relevant_chunks("огнетушитель", top_k=2)

        assert len(result) == 2
        assert result[0].document_id == 'doc1'
        assert result[0].document_title == 'СП 9.13130.2009'
        assert result[0].clause_text == 'Огнетушители должны размещаться'
        assert result[0].relevance_score == 0.9  # 1.0 - 0.1

        rag_service.collection.query.assert_called_once_with(
            query_texts=['огнетушитель'],
            n_results=2,
            include=['metadatas', 'documents', 'distances']
        )

    @pytest.mark.asyncio
    async def test_search_relevant_chunks_empty_result(self, rag_service):
        """Тест поиска с пустым результатом"""
        mock_results = {
            'ids': [[]],
            'metadatas': [[]],
            'documents': [[]],
            'distances': [[]]
        }

        rag_service.collection.query.return_value = mock_results

        result = await rag_service.search_relevant_chunks("несуществующий запрос")

        assert result == []

    @pytest.mark.asyncio
    async def test_search_relevant_chunks_exception(self, rag_service):
        """Тест обработки исключения при поиске"""
        rag_service.collection.query.side_effect = Exception("DB Error")

        result = await rag_service.search_relevant_chunks("запрос")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_document_info(self, rag_service):
        """Тест получения информации о документе"""
        chunk = RAGChunk(
            document_id="doc1",
            document_title="СП 9.13130.2009",
            document_number="9.13130.2009",
            clause_number="4.1.3",
            clause_text="Огнетушители должны размещаться",
            relevance_score=0.9
        )

        result = await rag_service.get_document_info(chunk)

        expected = {
            'document_title': 'СП 9.13130.2009',
            'document_number': '9.13130.2009',
            'clause_number': '4.1.3',
            'clause_text': 'Огнетушители должны размещаться'
        }

        assert result == expected
