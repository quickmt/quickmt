import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from quickmt.translator import Translator, TranslatorABC


# Mock objects
@pytest.fixture
def mock_ctranslate2():
    with patch("ctranslate2.Translator") as mock:
        yield mock


@pytest.fixture
def mock_sentencepiece():
    with patch("sentencepiece.SentencePieceProcessor") as mock:
        yield mock


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a dummy model directory with required files."""
    model_dir = tmp_path / "dummy-model"
    model_dir.mkdir()
    (model_dir / "src.spm.model").write_text("dummy")
    (model_dir / "tgt.spm.model").write_text("dummy")
    return model_dir


@pytest.fixture
def translator_instance(temp_model_dir, mock_ctranslate2, mock_sentencepiece):
    return Translator(temp_model_dir)


class TestTranslatorABC:
    def test_sentence_split(self):
        src = ["Hello world. This is a test.", "Another paragraph."]
        input_ids, paragraph_ids, sentences = TranslatorABC._sentence_split(src)

        assert len(sentences) == 3
        assert input_ids == [0, 0, 1]
        assert paragraph_ids == [0, 0, 0]
        assert sentences[0] == "Hello world."
        assert sentences[1] == "This is a test."
        assert sentences[2] == "Another paragraph."

    def test_sentence_join(self):
        input_ids = [0, 0, 1]
        paragraph_ids = [0, 0, 0]
        sentences = ["Hello world.", "This is a test.", "Another paragraph."]

        joined = TranslatorABC._sentence_join(input_ids, paragraph_ids, sentences)
        assert len(joined) == 2
        assert joined[0] == "Hello world. This is a test."
        assert joined[1] == "Another paragraph."

    def test_sentence_join_empty(self):
        assert TranslatorABC._sentence_join([], [], [], length=5) == [""] * 5


class TestTranslator:
    def test_init_joint_tokens(self, tmp_path, mock_ctranslate2, mock_sentencepiece):
        model_dir = tmp_path / "joint-model"
        model_dir.mkdir()
        (model_dir / "joint.spm.model").write_text("dummy")

        translator = Translator(model_dir)
        assert mock_sentencepiece.call_count == 2
        # Verify it used the joint model for both
        args, kwargs = mock_sentencepiece.call_args_list[0]
        assert "joint.spm.model" in kwargs["model_file"]

    def test_tokenize(self, translator_instance):
        translator_instance.source_tokenizer.encode.return_value = [
            ["token1", "token2"]
        ]
        result = translator_instance.tokenize(["Hello"])
        assert result == [["token1", "token2", "</s>"]]
        translator_instance.source_tokenizer.encode.assert_called_with(
            ["Hello"], out_type=str
        )

    def test_detokenize(self, translator_instance):
        translator_instance.target_tokenizer.decode.return_value = ["Hello"]
        result = translator_instance.detokenize([["token1", "token2"]])
        assert result == ["Hello"]
        translator_instance.target_tokenizer.decode.assert_called_with(
            [["token1", "token2"]]
        )

    def test_unload(self, translator_instance):
        del translator_instance.translator
        # Should not raise
        translator_instance.unload()

    def test_call_full_pipeline(self, translator_instance):
        # Mock the steps
        with (
            patch.object(Translator, "tokenize") as mock_tok,
            patch.object(Translator, "translate_batch") as mock_trans,
            patch.object(Translator, "detokenize") as mock_detok,
        ):
            mock_tok.return_value = [["tok"]]
            mock_res = MagicMock()
            mock_res.hypotheses = [["hypo"]]
            mock_trans.return_value = [mock_res]
            mock_detok.return_value = ["Translated sentence."]

            result = translator_instance("Source text.")
            assert result == "Translated sentence."

            mock_tok.assert_called_once()
            mock_trans.assert_called_once()
            mock_detok.assert_called_once()

    def test_translate_stream(self, translator_instance):
        translator_instance.translator.translate_iterable = MagicMock(
            return_value=[
                MagicMock(hypotheses=[["hypo1"]]),
                MagicMock(hypotheses=[["hypo2"]]),
            ]
        )

        with (
            patch.object(Translator, "tokenize") as mock_tok,
            patch.object(Translator, "detokenize") as mock_detok,
        ):
            mock_tok.return_value = [["tok1"], ["tok2"]]
            mock_detok.side_effect = lambda x: [f"Detok {x[0][0]}"]

            results = list(translator_instance.translate_stream(["Sent 1.", "Sent 2."]))
            assert len(results) == 2
            assert results[0]["translation"] == "Detok hypo1"
            assert results[1]["translation"] == "Detok hypo2"

    def test_translate_file(self, translator_instance, tmp_path):
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.txt"
        input_file.write_text("Line 1\nLine 2")

        with patch.object(Translator, "__call__") as mock_call:
            mock_call.return_value = ["Trans 1", "Trans 2"]
            translator_instance.translate_file(str(input_file), str(output_file))

            content = output_file.read_text()
            assert content == "Trans 1\nTrans 2\n"

    def test_translate_batch(self, translator_instance):
        translator_instance.translate_batch(
            [["tok"]],
            beam_size=10,
            patience=2,
            max_batch_size=16,
            num_hypotheses=5,  # kwargs
        )
        translator_instance.translator.translate_batch.assert_called_once()
        args, kwargs = translator_instance.translator.translate_batch.call_args
        assert kwargs["beam_size"] == 10
        assert kwargs["patience"] == 2
        assert kwargs["max_batch_size"] == 16
        assert kwargs["num_hypotheses"] == 5
