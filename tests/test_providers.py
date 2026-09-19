from backend.app.errors import ErrorCode
from backend.app.models import Chunk, RetrievedChunk
from backend.generation.base import GenerationContext, LLMProvider
from backend.generation.fake_provider import FakeLLMProvider
from backend.generation.gemini_provider import GeminiLLMProvider


def _make_context() -> GenerationContext:
    chunk_obj = Chunk(
        chunk_id="test_doc__p001__c0001",
        doc_id="sha123",
        document="test_doc.pdf",
        page=1,
        chunk_index=1,
        text="Sample text content.",
        char_count=20,
        token_estimate=5,
    )
    retrieved = RetrievedChunk(
        chunk=chunk_obj, score=0.88, rank=1, above_threshold=True
    )
    return GenerationContext(
        question="What is this test?",
        selected_chunks=[retrieved],
        formatted_prompt="Grounded prompt string",
        passages_count=1,
        context_chars=20,
        token_estimate=5,
    )


def test_fake_provider_deterministic_output():
    provider = FakeLLMProvider()

    assert provider.provider_name == "fake"
    assert provider.model_name == "fake-gemini-2.5-flash"
    assert provider.is_available() is True

    context = _make_context()
    res = provider.generate(context)

    assert res.status == "completed"
    assert res.provider == "fake"
    assert "What is this test?" in res.text
    assert "test_doc__p001__c0001" in res.text


def test_fake_provider_can_inject_invalid_citation():
    provider = FakeLLMProvider(inject_invalid_citation=True)
    context = _make_context()
    res = provider.generate(context)

    assert res.status == "completed"
    assert "nonexistent_chunk_id_999" in res.text


def test_gemini_provider_missing_key_behavior():
    # Instantiate GeminiLLMProvider with empty API key
    provider = GeminiLLMProvider(api_key="")

    assert provider.provider_name == "gemini"
    assert provider.model_name == "gemini-2.5-flash"
    assert provider.is_available() is False

    context = _make_context()
    res = provider.generate(context)

    assert res.status == "unavailable"
    assert res.text is None
    assert res.error_code == ErrorCode.PROVIDER_UNAVAILABLE
    assert "missing" in res.error_message.lower()


def test_protocol_satisfaction():
    fake_p: LLMProvider = FakeLLMProvider()
    gemini_p: LLMProvider = GeminiLLMProvider(api_key="")

    assert fake_p.provider_name == "fake"
    assert gemini_p.provider_name == "gemini"
