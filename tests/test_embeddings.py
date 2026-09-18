from __future__ import annotations

import numpy as np
import pytest

from embeddings.fake import FakeEmbedder
from embeddings.sentence_transformer import (
    BGESentenceTransformerEmbedder,
    get_sentence_transformer_model,
)


def test_fake_embedder_shape_and_normalization():
    embedder = FakeEmbedder(dimension=384)
    texts = ["Sample passage 1", "Sample passage 2", "Sample passage 3"]

    vectors = embedder.embed_chunks(texts)

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (3, 384)
    assert vectors.dtype == np.float32

    # Verify L2 normalization: norm of each row == 1.0 within float tolerance
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_fake_embedder_query_prefix_and_empty_input():
    embedder = FakeEmbedder(dimension=384)

    empty_vecs = embedder.embed_chunks([])
    assert empty_vecs.shape == (0, 384)

    query_vec = embedder.embed_query("What is edge computing?")
    assert query_vec.shape == (1, 384)
    query_norm = np.linalg.norm(query_vec)
    np.testing.assert_allclose(query_norm, 1.0, rtol=1e-5)


def test_fake_embedder_preserves_ordering():
    embedder = FakeEmbedder()
    texts = ["Alpha text", "Beta text", "Gamma text"]

    vecs = embedder.embed_chunks(texts)
    vec_alpha = embedder.embed_chunks(["Alpha text"])
    vec_beta = embedder.embed_chunks(["Beta text"])

    np.testing.assert_array_equal(vecs[0], vec_alpha[0])
    np.testing.assert_array_equal(vecs[1], vec_beta[0])


@pytest.mark.integration
def test_bge_sentence_transformer_embedder_integration():
    """Integration test for real BGE embedder shape, dimension, normalization."""
    embedder = BGESentenceTransformerEmbedder()
    assert embedder.dimension == 384
    assert embedder.model_name == "BAAI/bge-small-en-v1.5"

    texts = ["Edge computing reduces latency.", "Predictive maintenance."]
    vectors = embedder.embed_chunks(texts)

    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-4)

    # Test singleton model instance reuse
    model1 = get_sentence_transformer_model()
    model2 = get_sentence_transformer_model()
    assert model1 is model2


@pytest.mark.integration
def test_bge_query_prefix_behavior():
    """Verify BGE query prefix instruction is prepended to queries."""
    embedder = BGESentenceTransformerEmbedder()
    query_str = "What is edge computing?"

    # Query embedding uses BGE prefix
    query_vec = embedder.embed_query(query_str)
    # Chunk embedding embeds raw text without prefix
    chunk_vec = embedder.embed_chunks([query_str])

    assert query_vec.shape == (1, 384)
    assert chunk_vec.shape == (1, 384)
    # Query vector and raw chunk vector must differ due to BGE query instruction prefix
    assert not np.allclose(query_vec, chunk_vec, atol=1e-3)

