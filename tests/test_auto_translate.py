import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_auto_detect_src_lang(client: AsyncClient):
    """Verify that src_lang is auto-detected if missing."""
    # Ensure some models are available
    models_res = await client.get("/api/models")
    available_models = models_res.json()["models"]
    if not any(
        m["src_lang"] == "fr" and m["tgt_lang"] == "en" for m in available_models
    ):
        pytest.skip("fr-en model needed for this test")

    payload = {"src": "Bonjour tout le monde", "tgt_lang": "en"}
    response = await client.post("/api/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "translation" in data
    assert data["src_lang"] == "fr"
    assert 0.0 < data["src_lang_score"] <= 1.0
    assert data["tgt_lang"] == "en"
    assert "quickmt/quickmt-fr-en" in data["model_used"]


@pytest.mark.asyncio
async def test_default_tgt_lang(client: AsyncClient):
    """Verify that tgt_lang defaults to 'en'."""
    models_res = await client.get("/api/models")
    available_models = models_res.json()["models"]
    if not any(
        m["src_lang"] == "fr" and m["tgt_lang"] == "en" for m in available_models
    ):
        pytest.skip("fr-en model needed for this test")

    payload = {"src": "Bonjour", "src_lang": "fr"}
    response = await client.post("/api/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["src_lang"] == "fr"
    assert data["src_lang_score"] == 1.0
    assert data["tgt_lang"] == "en"
    assert "quickmt/quickmt-fr-en" in data["model_used"]


@pytest.mark.asyncio
async def test_mixed_language_batch(client: AsyncClient):
    """Verify that a batch with mixed languages is handled correctly."""
    models_res = await client.get("/api/models")
    available_models = models_res.json()["models"]

    needed = [("fr", "en"), ("es", "en")]
    for src, tgt in needed:
        if not any(
            m["src_lang"] == src and m["tgt_lang"] == tgt for m in available_models
        ):
            pytest.skip(f"Mixed batch test needs both fr-en and es-en models")

    payload = {"src": ["Bonjour tout le monde", "Hola amigos"], "tgt_lang": "en"}
    response = await client.post("/api/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["translation"], list)
    assert len(data["translation"]) == 2
    assert data["src_lang"] == ["fr", "es"]
    assert len(data["src_lang_score"]) == 2
    assert all(0.0 < s <= 1.0 for s in data["src_lang_score"])
    assert data["tgt_lang"] == "en"
    assert "quickmt/quickmt-fr-en" in data["model_used"]
    assert "quickmt/quickmt-es-en" in data["model_used"]


@pytest.mark.asyncio
async def test_identity_translation(client: AsyncClient):
    """Verify that translation is skipped if src_lang == tgt_lang."""
    payload = {"src": "This is already English", "src_lang": "en", "tgt_lang": "en"}
    response = await client.post("/api/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["translation"] == "This is already English"
    assert data["src_lang"] == "en"
    assert data["src_lang_score"] == 1.0
    assert data["tgt_lang"] == "en"
    assert data["model_used"] == "identity"


@pytest.mark.asyncio
async def test_auto_detect_mixed_identity(client: AsyncClient):
    """Verify mixed batch with some items needing translation and some remaining as-is."""
    models_res = await client.get("/api/models")
    available_models = models_res.json()["models"]
    if not any(
        m["src_lang"] == "fr" and m["tgt_lang"] == "en" for m in available_models
    ):
        pytest.skip("fr-en model needed for this test")

    payload = {"src": ["Bonjour", "Hello world"], "tgt_lang": "en"}
    response = await client.post("/api/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["translation"]) == 2
    assert data["src_lang"] == ["fr", "en"]
    assert len(data["src_lang_score"]) == 2
    # First should be auto-detected, second should be auto-detected (and high confidence)
    assert all(0.0 < s <= 1.0 for s in data["src_lang_score"])
    assert data["tgt_lang"] == "en"
    assert data["model_used"] == ["quickmt/quickmt-fr-en", "identity"]
