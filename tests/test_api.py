import pytest
import asyncio
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "loaded_models" in data


@pytest.mark.asyncio
async def test_get_models(client: AsyncClient):
    response = await client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)


@pytest.mark.asyncio
async def test_get_languages(client: AsyncClient):
    response = await client.get("/api/languages")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    # Check structure if models exist
    if data:
        src = list(data.keys())[0]
        assert isinstance(data[src], list)


@pytest.mark.asyncio
async def test_translate_single(client: AsyncClient):
    # First, find an available model
    models_res = await client.get("/api/models")
    models = models_res.json()["models"]
    if not models:
        pytest.skip("No models available in MODELS_DIR")

    model = models[0]
    payload = {
        "src": "Hello world",
        "src_lang": model["src_lang"],
        "tgt_lang": model["tgt_lang"],
    }

    response = await client.post("/api/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "translation" in data
    assert "processing_time" in data
    assert data["src_lang"] == model["src_lang"]
    assert data["src_lang_score"] == 1.0
    assert data["tgt_lang"] == model["tgt_lang"]
    assert data["model_used"] == model["model_id"]


@pytest.mark.asyncio
async def test_translate_list(client: AsyncClient):
    models_res = await client.get("/api/models")
    models = models_res.json()["models"]
    if not models:
        pytest.skip("No models available")

    model = models[0]
    payload = {
        "src": ["Hello", "World"],
        "src_lang": model["src_lang"],
        "tgt_lang": model["tgt_lang"],
    }

    response = await client.post("/api/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["translation"], list)
    assert len(data["translation"]) == 2
    assert data["src_lang"] == [model["src_lang"], model["src_lang"]]
    assert data["src_lang_score"] == [1.0, 1.0]
    assert data["tgt_lang"] == model["tgt_lang"]
    assert data["model_used"] == [model["model_id"], model["model_id"]]


@pytest.mark.asyncio
async def test_dynamic_batching(client: AsyncClient):
    """Verify that multiple concurrent requests work correctly (triggering batching logic)."""
    models_res = await client.get("/api/models")
    models = models_res.json()["models"]
    if not models:
        pytest.skip("No models available")

    model = models[0]
    src, tgt = model["src_lang"], model["tgt_lang"]

    texts = [f"Sentence number {i}" for i in range(5)]
    tasks = []

    for text in texts:
        payload = {"src": text, "src_lang": src, "tgt_lang": tgt}
        tasks.append(client.post("/api/translate", json=payload))

    responses = await asyncio.gather(*tasks)

    for response in responses:
        assert response.status_code == 200
        assert "translation" in response.json()
