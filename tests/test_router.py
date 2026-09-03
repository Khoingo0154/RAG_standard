from unittest.mock import MagicMock, patch
import pytest

from retrieval.router import QueryIntent, QueryRouter
from retrieval.pipeline import search_and_generate


def test_router_empty_query_returns_chitchat():
    router = QueryRouter()
    assert router.classify("") == QueryIntent.CHITCHAT
    assert router.classify("   ") == QueryIntent.CHITCHAT


def test_router_no_api_key_fallbacks_to_document_query():
    router = QueryRouter(gemini_api_key="")
    assert router.classify("Việt vị là gì?") == QueryIntent.DOCUMENT_QUERY


@patch("google.genai.Client")
def test_router_classifies_chitchat(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.models.generate_content.return_value = MagicMock(text="CHITCHAT")

    router = QueryRouter(gemini_api_key="fake-key")
    assert router.classify("Xin chào") == QueryIntent.CHITCHAT


@patch("google.genai.Client")
def test_router_classifies_out_of_scope(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.models.generate_content.return_value = MagicMock(text="OUT_OF_SCOPE")

    router = QueryRouter(gemini_api_key="fake-key")
    assert router.classify("Cách nấu phở?") == QueryIntent.OUT_OF_SCOPE


@patch("google.genai.Client")
def test_router_classifies_document_query(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.models.generate_content.return_value = MagicMock(text="DOCUMENT_QUERY")

    router = QueryRouter(gemini_api_key="fake-key")
    assert router.classify("Việt vị là gì?") == QueryIntent.DOCUMENT_QUERY


def test_pipeline_handles_chitchat_without_retriever():
    mock_router = MagicMock()
    mock_router.classify.return_value = QueryIntent.CHITCHAT

    with patch("retrieval.pipeline.Generator") as mock_gen_cls:
        mock_gen = MagicMock()
        mock_gen_cls.return_value = mock_gen
        mock_gen.generate_chitchat.return_value = "Xin chào! Tôi có thể giúp gì cho bạn về Luật FIFA?"

        response = search_and_generate(
            query="Xin chào",
            router=mock_router,
        )

        assert response.intent == "CHITCHAT"
        assert response.sources == []
        assert "Xin chào" in response.answer


def test_pipeline_handles_out_of_scope_directly():
    mock_router = MagicMock()
    mock_router.classify.return_value = QueryIntent.OUT_OF_SCOPE

    response = search_and_generate(
        query="Thời tiết Hà Nội?",
        router=mock_router,
    )

    assert response.intent == "OUT_OF_SCOPE"
    assert response.sources == []
    assert "chuyên môn về tài liệu Luật bóng đá FIFA" in response.answer
