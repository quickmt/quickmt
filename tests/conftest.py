import pytest
import os
from typing import AsyncGenerator
from quickmt.rest_server import app
from httpx import AsyncClient


@pytest.fixture(scope="session")
def base_url() -> str:
    return os.getenv("TEST_BASE_URL", "http://127.0.0.1:8000")


@pytest.fixture
async def client(base_url: str) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(base_url=base_url, timeout=60.0) as client:
        yield client
