import asyncio
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer
from models.violation import RAGChunk
from config import settings
import numpy as np


class RAGService:
    """Сервис для работы с RAG базой данных ChromaDB"""

    def __init__(self):
        self.db_path = settings.rag_database_path
        self.model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )

        print(f"Инициализация ChromaDB с путем: {self.db_path}")

        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            print("ChromaDB клиент создан успешно")
        except Exception as e:
            print(f"Ошибка создания ChromaDB клиента: {e}")
            raise

        # Используем get_or_create_collection для безопасного получения коллекции
        try:
            self.collection = self.client.get_or_create_collection(
                name="pipeline_standards")
            print("Коллекция 'pipeline_standards' получена/создана успешно")
        except Exception as e:
            print(f"Ошибка при инициализации ChromaDB: {e}")
            # Fallback - пытаемся получить существующую коллекцию
            try:
                self.collection = self.client.get_collection(
                    name="pipeline_standards")
                print("Коллекция 'pipeline_standards' получена через fallback")
            except Exception as e2:
                print(f"Fallback тоже не сработал: {e2}")
                # Если ничего не работает, создаем новую коллекцию
                try:
                    self.collection = self.client.create_collection(
                        name="pipeline_standards")
                    print("Коллекция 'pipeline_standards' создана заново")
                except Exception as e3:
                    print(f"Не удалось создать коллекцию: {e3}")
                    raise

    async def search_relevant_chunks(self, query: str, top_k: int = 3) -> List[RAGChunk]:
        """
        Ищет релевантные чанки в базе данных ChromaDB

        Args:
            query: Поисковый запрос
            top_k: Количество результатов

        Returns:
            List[RAGChunk]: Список релевантных чанков
        """
        try:
            # Используем ChromaDB API для поиска
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=['metadatas', 'documents', 'distances']
            )

            chunks = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {
                    }
                    document = results['documents'][0][i] if results['documents'] and results['documents'][0] else ""
                    distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.0

                    chunk = RAGChunk(
                        document_id=doc_id,
                        document_title=metadata.get('document_title', ''),
                        document_number=metadata.get('document_number', ''),
                        clause_number=metadata.get('clause_number', ''),
                        clause_text=document,
                        keywords=metadata.get('keywords', ''),
                        relevance_score=1.0 - distance  # Конвертируем расстояние в схожесть
                    )
                    chunks.append(chunk)

            print(f"[RAGService] Найдено чанков: {len(chunks)}")
            for i, ch in enumerate(chunks, 1):
                print(f"  Чанк {i}: {ch}")

            return chunks

        except Exception as e:
            print(f"Ошибка при поиске в ChromaDB: {e}")
            return []

    async def get_document_info(self, chunk: RAGChunk) -> dict:
        """
        Получает дополнительную информацию о документе

        Args:
            chunk: RAG чанк

        Returns:
            dict: Информация о документе
        """
        return {
            'document_title': chunk.document_title,
            'document_number': chunk.document_number,
            'clause_number': chunk.clause_number,
            'clause_text': chunk.clause_text
        }
