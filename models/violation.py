from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class Violation(BaseModel):
    """Модель данных нарушения"""
    original_text: str = Field(...,
                               description="Исходный текст из голосового сообщения")
    corrected_description: Optional[str] = Field(
        None, description="Скорректированное описание нарушения")
    document_title: Optional[str] = Field(
        None, description="Название нормативного документа")
    document_number: Optional[str] = Field(None, description="Номер документа")
    clause_number: Optional[str] = Field(None, description="Номер пункта")
    suggestions: Optional[str] = Field(
        None, description="Предлагаемые меры по устранению")
    chunks: List[dict] = Field(
        default_factory=list, description="Релевантные чанки из RAG базы")
    error: Optional[str] = Field(
        None, description="Ошибка обработки, если есть")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Время обработки")


class ViolationResponse(BaseModel):
    """Модель ответа бота"""
    corrected_description: str = Field(...,
                                       description="Скорректированное описание нарушения")
    document_info: str = Field(...,
                               description="Информация о нормативном документе")
    suggestions: str = Field(...,
                             description="Предлагаемые меры по устранению")
    success: bool = Field(..., description="Успешность обработки")
    error_message: Optional[str] = Field(
        None, description="Сообщение об ошибке")


class RAGChunk(BaseModel):
    """Модель чанка из RAG базы"""
    document_id: str = Field(..., description="ID документа")
    document_title: str = Field(..., description="Название документа")
    document_number: str = Field(..., description="Номер документа")
    clause_number: str = Field(..., description="Номер пункта")
    clause_text: str = Field(..., description="Текст пункта")
    relevance_score: float = Field(..., description="Оценка релевантности")
