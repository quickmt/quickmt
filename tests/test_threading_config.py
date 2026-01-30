import pytest
from unittest.mock import patch
from quickmt.manager import ModelManager


@pytest.mark.asyncio
async def test_threading_config_propagation():
    """Verify that inter_threads and intra_threads are passed to CTranslate2."""

    # Mocking components to prevent actual model loading
    with patch("quickmt.manager.Translator") as mock_translator_cls:
        # Configuration
        inter = 2
        intra = 4

        manager = ModelManager(
            max_loaded=1,
            device="cpu",
            compute_type="int8",
            inter_threads=inter,
            intra_threads=intra,
        )

        # Inject a dummy model to collection
        manager.hf_collection_models = [
            {"model_id": "test/model", "src_lang": "en", "tgt_lang": "fr"}
        ]

        # Mock snapshot_download
        with patch("quickmt.manager.snapshot_download", return_value="/tmp/model"):
            # Trigger model load
            await manager.get_model("en", "fr")

            # Verify Translator was instantiated with correct parameters
            args, kwargs = mock_translator_cls.call_args
            assert kwargs["inter_threads"] == inter
            assert kwargs["intra_threads"] == intra
            assert kwargs["device"] == "cpu"
            assert kwargs["compute_type"] == "int8"
