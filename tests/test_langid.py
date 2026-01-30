import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from quickmt.langid import LanguageIdentification

@pytest.fixture
def mock_fasttext():
    with patch("fasttext.load_model") as mock_load:
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        # Configure default behavior for predict
        def mock_predict(items, k=1, threshold=0.0):
            # Return ([['__label__en', ...]], [[0.9, ...]])
            labels = [["__label__en"] * k for _ in items]
            scores = [[0.9] * k for _ in items]
            return labels, scores
            
        mock_model.predict.side_effect = mock_predict
        yield mock_model

@pytest.fixture
def langid_model(mock_fasttext, tmp_path):
    # Create a dummy model file so the existence check passes
    model_path = tmp_path / "model.bin"
    model_path.write_text("dummy content")
    return LanguageIdentification(model_path)


def test_predict_single(langid_model, mock_fasttext):
    result = langid_model.predict("Hello world")
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == ("en", 0.9)
    mock_fasttext.predict.assert_called_once_with(["Hello world"], k=1, threshold=0.0)

def test_predict_batch(langid_model, mock_fasttext):
    texts = ["Hello", "Bonjour"]
    results = langid_model.predict(texts, k=2)
    
    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert len(r) == 2
        assert r[0] == ("en", 0.9)
    
    mock_fasttext.predict.assert_called_once_with(texts, k=2, threshold=0.0)

def test_predict_best_single(langid_model):
    result = langid_model.predict_best("Hello")
    assert result == "en"

def test_predict_best_batch(langid_model):
    results = langid_model.predict_best(["Hello", "World"])
    assert results == ["en", "en"]

def test_predict_threshold(langid_model, mock_fasttext):
    # Configure mock to return nothing if threshold is high (simulated)
    def mock_predict_low_score(items, k=1, threshold=0.0):
        if threshold > 0.9:
            return [[] for _ in items], [[] for _ in items]
        return [["__label__en"] for _ in items], [[0.9] for _ in items]
        
    mock_fasttext.predict.side_effect = mock_predict_low_score
    
    result = langid_model.predict_best("Hello", threshold=0.95)
    assert result is None
    
    result = langid_model.predict_best("Hello", threshold=0.5)
    assert result == "en"
