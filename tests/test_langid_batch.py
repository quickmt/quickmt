import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_langid_batch(client: AsyncClient):
    """Verify that language identification works for a list of strings."""
    payload = {"src": ["This is English text.", "Ceci est un texte français."]}
    response = await client.post("/api/identify-language", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Expect a list of lists of DetectionResult
    results = data["results"]
    assert isinstance(results, list)
    assert len(results) == 2

    # First item: English
    assert len(results[0]) >= 1
    assert results[0][0]["lang"] == "en"

    # Second item: French
    assert len(results[1]) >= 1
    assert results[1][0]["lang"] == "fr"


@pytest.mark.asyncio
async def test_langid_newline_handling(client: AsyncClient):
    """Verify that inputs with newlines are handled gracefully (no 500 error)."""
    # Single string with newline
    payload_single = {"src": "This text\nhas a newline."}
    response = await client.post("/api/identify-language", json=payload_single)
    assert response.status_code == 200
    data = response.json()
    assert data["results"][0]["lang"] == "en"

    # Batch with newlines
    payload_batch = {"src": ["Line 1\nLine 2", "Another\nline"]}
    response = await client.post("/api/identify-language", json=payload_batch)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
