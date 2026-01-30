import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_explicit_mixed_languages(client: AsyncClient):
    """Verify explicit src_lang list for a mixed batch."""
    # Ensure needed models are available
    models_res = await client.get("/api/models")
    available_models = models_res.json()["models"]

    needed = [("fr", "en"), ("es", "en")]
    for src, tgt in needed:
        if not any(
            m["src_lang"] == src and m["tgt_lang"] == tgt for m in available_models
        ):
            pytest.skip(f"Mixed batch test needs both fr-en and es-en models")

    # Explicitly specify languages for each input
    payload = {"src": ["Bonjour", "Hola"], "src_lang": ["fr", "es"], "tgt_lang": "en"}

    response = await client.post("/api/translate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["src_lang"] == ["fr", "es"]
    assert data["model_used"] == ["quickmt/quickmt-fr-en", "quickmt/quickmt-es-en"]
    assert len(data["translation"]) == 2


@pytest.mark.asyncio
async def test_src_lang_length_mismatch(client: AsyncClient):
    """Verify 422 error when src and src_lang lengths differ."""
    payload = {
        "src": ["Hello", "World"],
        "src_lang": ["en"],  # Only 1 language for 2 inputs
        "tgt_lang": "es",
    }

    response = await client.post("/api/translate", json=payload)
    assert response.status_code == 422
    assert (
        "src_lang list length must match src list length" in response.json()["detail"]
    )


@pytest.mark.asyncio
async def test_src_lang_list_with_single_src(client: AsyncClient):
    """Verify single src string with single-item src_lang list is not allowed or handled gracefully."""
    # The Pydantic model allows this, but our logic checks lengths.
    # If src is str, src_list has len 1. If src_lang is list, it must have len 1.

    # Needs a model
    models_res = await client.get("/api/models")
    models = models_res.json()["models"]
    if not models:
        pytest.skip("No models available")
    model = models[0]

    payload = {
        "src": "Hello",
        "src_lang": [model["src_lang"]],
        "tgt_lang": model["tgt_lang"],
    }

    response = await client.post("/api/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["src_lang"] == model["src_lang"]
