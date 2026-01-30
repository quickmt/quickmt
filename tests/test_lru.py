import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_lru_eviction(client: AsyncClient):
    """
    Test that the server correctly unloads the least recently used model
    when MAX_LOADED_MODELS is exceeded.
    """
    # 1. Get available models
    models_res = await client.get("/api/models")
    available_models = models_res.json()["models"]

    # 2. Get MAX_LOADED_MODELS from health
    health_res = await client.get("/api/health")
    max_models = health_res.json()["max_models"]

    if len(available_models) <= max_models:
        pytest.skip(
            f"Not enough models in MODELS_DIR to test eviction (need > {max_models})"
        )

    # 3. Load max_models + 1 models sequentially
    loaded_in_order = []
    for i in range(max_models + 1):
        model = available_models[i]
        payload = {
            "src": "test",
            "src_lang": model["src_lang"],
            "tgt_lang": model["tgt_lang"],
        }
        await client.post("/api/translate", json=payload)
        loaded_in_order.append(f"{model['src_lang']}-{model['tgt_lang']}")

    # 4. Check currently loaded models
    health_after = await client.get("/api/health")
    currently_loaded = health_after.json()["loaded_models"]

    # The first model should have been evicted
    first_model = loaded_in_order[0]
    assert first_model not in currently_loaded
    assert len(currently_loaded) == max_models

    # The most recently requested model should be there
    last_model = loaded_in_order[-1]
    assert last_model in currently_loaded
