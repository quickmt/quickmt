import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_identify_language_single(client: AsyncClient):
    """Verify single string language identification."""
    payload = {"src": "Hello, how are you?", "k": 1}
    response = await client.post("/api/identify-language", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "processing_time" in data
    assert isinstance(data["results"], list)
    # FastText should identify this as English
    assert data["results"][0]["lang"] == "en"


@pytest.mark.asyncio
async def test_identify_language_batch(client: AsyncClient):
    """Verify batch language identification."""
    payload = {"src": ["Bonjour tout le monde", "Hola amigos"], "k": 1}
    response = await client.post("/api/identify-language", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    assert data["results"][0][0]["lang"] == "fr"
    assert data["results"][1][0]["lang"] == "es"


@pytest.mark.asyncio
async def test_identify_language_threshold(client: AsyncClient):
    """Verify threshold filtering in the endpoint."""
    payload = {"src": "This is definitely English", "k": 5, "threshold": 0.9}
    response = await client.post("/api/identify-language", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Only 'en' should probably be above 0.9
    assert len(data["results"]) == 1
    assert data["results"][0]["lang"] == "en"
