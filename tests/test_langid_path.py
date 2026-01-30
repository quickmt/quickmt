from pathlib import Path
import os
from unittest.mock import patch
from quickmt.langid import ensure_model_exists, LanguageIdentification

def test_langid_default_path():
    """Verify that LanguageIdentification uses the XDG cache path by default."""
    # Mock os.getenv to ensure we test the default behavior, but respect XDG_CACHE_HOME if we want to mock it.
    # Here we simulate no explicit model path provided.
    
    with patch("quickmt.langid.fasttext.load_model") as mock_load, \
         patch("quickmt.langid.urllib.request.urlretrieve") as mock_retrieve, \
         patch("pathlib.Path.exists") as mock_exists, \
         patch("pathlib.Path.mkdir") as mock_mkdir:
        
        # Simulate model cached and exists
        mock_exists.return_value = True 
        
        lid = LanguageIdentification(model_path=None)
        
        # Verify load_model was called with a path in the cache
        args, _ = mock_load.call_args
        loaded_path = str(args[0])
        
        expected_part = os.path.join(".cache", "fasttext_language_id", "lid.176.bin")
        assert expected_part in loaded_path

        # Old path should not be used
        assert "models/lid.176.ftz" not in loaded_path

def test_ensure_model_exists_path():
    """Verify ensure_model_exists resolves to cache path."""
    with patch("quickmt.langid.urllib.request.urlretrieve") as mock_retrieve, \
         patch("pathlib.Path.exists") as mock_exists, \
         patch("pathlib.Path.mkdir") as mock_mkdir:
         
        # Simulate model missing to trigger download logic path check
        mock_exists.return_value = False
        
        ensure_model_exists(None)
        
        # Check download target
        args, _ = mock_retrieve.call_args
        download_target = str(args[1])
        
        expected_part = os.path.join(".cache", "fasttext_language_id", "lid.176.bin")
        assert expected_part in download_target
