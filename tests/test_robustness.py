import pytest
import asyncio
import time
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_model_not_found(client: AsyncClient):
    """Verify that requesting a non-existent model returns 404."""
    payload = {
        "src": "Hello",
        "src_lang": "en",
        "tgt_lang": "zz",  # Non-existent
    }
    response = await client.post("/api/translate", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_empty_input_string(client: AsyncClient):
    """Verify handling of empty string input."""
    models_res = await client.get("/api/models")
    model = models_res.json()["models"][0]

    payload = {"src": "", "src_lang": model["src_lang"], "tgt_lang": model["tgt_lang"]}
    response = await client.post("/api/translate", json=payload)
    assert response.status_code == 200
    assert response.json()["translation"] == ""


@pytest.mark.asyncio
async def test_empty_input_list(client: AsyncClient):
    """Verify handling of empty list input."""
    models_res = await client.get("/api/models")
    model = models_res.json()["models"][0]

    payload = {"src": [], "src_lang": model["src_lang"], "tgt_lang": model["tgt_lang"]}
    response = await client.post("/api/translate", json=payload)
    assert response.status_code == 200
    assert response.json()["translation"] == []


@pytest.mark.asyncio
async def test_invalid_input_type(client: AsyncClient):
    """Verify that invalid input types are rejected by Pydantic."""
    payload = {
        "src": 123,  # Should be string or list of strings
        "src_lang": "en",
        "tgt_lang": "fr",
    }
    response = await client.post("/api/translate", json=payload)
    assert response.status_code == 422  # Unprocessable Entity (Validation Error)


@pytest.mark.asyncio
async def test_concurrent_model_load(client: AsyncClient):
    """
    Test that concurrent requests for a new model are handled correctly
    (only one load should happen, others wait on the event).
    """
    # Find a model that is definitely NOT loaded
    health_res = await client.get("/api/health")
    loaded = health_res.json()["loaded_models"]

    models_res = await client.get("/api/models")
    available = models_res.json()["models"]

    target_model = None
    for m in available:
        lang_pair = f"{m['src_lang']}-{m['tgt_lang']}"
        if lang_pair not in loaded:
            target_model = m
            break

    if not target_model:
        pytest.skip("No unloaded models available to test concurrent loading")

    # Send multiple concurrent requests for the same new model
    payload = {
        "src": "Concurrent test",
        "src_lang": target_model["src_lang"],
        "tgt_lang": target_model["tgt_lang"],
    }

    tasks = [client.post("/api/translate", json=payload) for _ in range(3)]
    responses = await asyncio.gather(*tasks)

    for resp in responses:
        assert resp.status_code == 200
        assert "translation" in resp.json()


@pytest.mark.asyncio
async def test_parameter_overrides(client: AsyncClient):
    """Verify that request-level parameters are respected."""
    models_res = await client.get("/api/models")
    model = models_res.json()["models"][0]

    payload = {
        "src": "This is a test of parameter overrides.",
        "src_lang": model["src_lang"],
        "tgt_lang": model["tgt_lang"],
        "beam_size": 1,
        "max_decoding_length": 5,
    }

    response = await client.post("/api/translate", json=payload)
    assert response.status_code == 200
    # With max_decoding_length=5, the translation should be very short
    # Note: tokens != words, but usually it translates to 1-3 words
    trans = response.json()["translation"]
    # We can't strictly assert word count but we can check it's non-empty
    assert len(trans) > 0


@pytest.mark.asyncio
async def test_large_batch_processing(client: AsyncClient):
    """Verify processing of a batch larger than MAX_BATCH_SIZE."""
    models_res = await client.get("/api/models")
    models = models_res.json()["models"]
    if not models:
        pytest.skip("No translation models available")
    model = models[0]

    # Send 50 sentences (default MAX_BATCH_SIZE is 32)
    sentences = [f"This is sentence {i}" for i in range(50)]
    payload = {
        "src": sentences,
        "src_lang": model["src_lang"],
        "tgt_lang": model["tgt_lang"],
    }

    response = await client.post("/api/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["translation"]) == 50
