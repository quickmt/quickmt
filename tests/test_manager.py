import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from quickmt.manager import ModelManager, BatchTranslator
from quickmt.translator import Translator

@pytest.fixture
def mock_translator():
    with patch("quickmt.manager.Translator") as mock:
        instance = MagicMock()
        mock.return_value = instance
        yield instance

@pytest.fixture
def mock_hf():
    with patch("quickmt.manager.snapshot_download") as mock_dl, \
         patch("quickmt.manager.HfApi") as mock_api:
        
        # Mock collection fetch
        coll = MagicMock()
        coll.items = [
            MagicMock(item_id="quickmt/quickmt-en-fr", item_type="model"),
            MagicMock(item_id="quickmt/quickmt-fr-en", item_type="model")
        ]
        mock_api.return_value.get_collection.return_value = coll
        mock_dl.return_value = "/tmp/mock-model-path"
        
        yield mock_api, mock_dl

class TestBatchTranslator:
    @pytest.mark.asyncio
    async def test_translate_single(self, mock_translator):
        bt = BatchTranslator("test-id", "/tmp/path")
        
        # Mock translator call
        mock_translator.return_value = "Hola"
        
        result = await bt.translate("Hello", src_lang="en", tgt_lang="es")
        assert result == "Hola"
        assert bt.worker_task is not None
        
        await bt.stop_worker()
        assert bt.worker_task is None

class TestModelManager:
    @pytest.mark.asyncio
    async def test_fetch_hf_models(self, mock_hf):
        mm = ModelManager(max_loaded=2, device="cpu")
        await mm.fetch_hf_models()
        
        assert len(mm.hf_collection_models) == 2
        assert mm.hf_collection_models[0]["src_lang"] == "en"
        assert mm.hf_collection_models[0]["tgt_lang"] == "fr"

    @pytest.mark.asyncio
    async def test_get_model_lazy_load(self, mock_hf, mock_translator):
        mm = ModelManager(max_loaded=2, device="cpu")
        await mm.fetch_hf_models()
        
        # This should trigger download and start worker
        bt = await mm.get_model("en", "fr")
        assert isinstance(bt, BatchTranslator)
        assert "en-fr" in mm.models
        assert bt.model_id == "quickmt/quickmt-en-fr"

    @pytest.mark.asyncio
    async def test_lru_eviction(self, mock_hf, mock_translator):
        # Set max_loaded to 1 to trigger eviction immediately
        mm = ModelManager(max_loaded=1, device="cpu")
        await mm.fetch_hf_models()
        
        # Load first
        bt1 = await mm.get_model("en", "fr")
        assert len(mm.models) == 1
        
        # Load second (should evict first)
        bt2 = await mm.get_model("fr", "en")
        assert len(mm.models) == 1
        assert "fr-en" in mm.models
        assert "en-fr" not in mm.models
    @pytest.mark.asyncio
    async def test_get_model_cache_first(self, mock_hf, mock_translator):
        mock_api, mock_dl = mock_hf
        mm = ModelManager(max_loaded=2, device="cpu")
        await mm.fetch_hf_models()
        
        # Scenario 1: Local cache hit
        # Reset mock to track new calls
        mock_dl.reset_mock()
        mock_dl.return_value = "/tmp/mock-model-path"
        
        await mm.get_model("en", "fr")
        
        # Verify it tried local_files_only=True first
        assert mock_dl.call_count == 1
        args, kwargs = mock_dl.call_args
        assert kwargs.get("local_files_only") is True

    @pytest.mark.asyncio
    async def test_get_model_fallback(self, mock_hf, mock_translator):
        mock_api, mock_dl = mock_hf
        mm = ModelManager(max_loaded=2, device="cpu")
        await mm.fetch_hf_models()
        
        # Scenario 2: Local cache miss, fallback to online
        # First call fails, second succeeds
        mock_dl.side_effect = [Exception("Not found locally"), "/tmp/mock-model-path"]
        
        await mm.get_model("fr", "en")
        
        assert mock_dl.call_count == 2
        # First call was local only
        args1, kwargs1 = mock_dl.call_args_list[0]
        assert kwargs1.get("local_files_only") is True
        # Second call was online (no local_files_only or False)
        args2, kwargs2 = mock_dl.call_args_list[1]
        assert not kwargs2.get("local_files_only")
