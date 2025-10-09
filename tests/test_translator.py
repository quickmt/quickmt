from pathlib import Path

import pytest

from quickmt import (M2m100Translator, NllbTranslator, OpusmtTranslator,
                     Translator)

quickmt_model_path = "../quickmt-models/quickmt-fr-en/"

assert Path(quickmt_model_path).exists()


@pytest.fixture
def qmt_translator():
    return Translator(model_path=quickmt_model_path, device="cpu")


def test_translate_string(qmt_translator):
    ret = qmt_translator("C'est la vie")
    assert isinstance(ret, str)
    assert len(ret) > 0
    assert len(ret) < 100


def test_translate_list(qmt_translator):
    ret = qmt_translator(["C'est la vie", "C'est la vie, n'est pas?"])
    assert isinstance(ret, list)
    for i in ret:
        assert isinstance(i, str)
    assert len(ret) == 2


def test_tokenize_source(qmt_translator):
    ret = qmt_translator.tokenize("C'est la vie")
    assert isinstance(ret, list)
    for i in ret:
        assert isinstance(i, str)

    ret = qmt_translator.tokenize(["C'est la vie"])
    assert isinstance(ret, list)
    for i in ret:
        assert isinstance(i, list)
        for j in i:
            assert isinstance(j, str)


def test_tokenize_target(qmt_translator):
    ret = qmt_translator.tokenize("C'est la vie", source=False)
    assert isinstance(ret, list)
    for i in ret:
        assert isinstance(i, str)

    ret = qmt_translator.tokenize(["C'est la vie"], source=False)
    assert isinstance(ret, list)
    for i in ret:
        assert isinstance(i, list)
        for j in i:
            assert isinstance(j, str)


def test_detokenize_source(qmt_translator):
    ret = qmt_translator.detokenize(
        ["▁C", "'", "est", "▁la", "▁vie", "."], target=False
    )
    assert isinstance(ret, str)

    ret = qmt_translator.detokenize(
        [["▁C", "'", "est", "▁la", "▁vie", "."]], target=False
    )
    assert isinstance(ret, list)
    for i in ret:
        assert isinstance(i, str)


def test_detokenize_target(qmt_translator):
    ret = qmt_translator.detokenize(["▁C", "'", "est", "▁la", "▁vie", "."])
    assert isinstance(ret, str)

    ret = qmt_translator.detokenize([["▁C", "'", "est", "▁la", "▁vie", "."]])
    assert isinstance(ret, list)
    for i in ret:
        assert isinstance(i, str)


def test_sentence_split(qmt_translator):
    in_ids, p_ids, sents = qmt_translator._sentence_split(
        [
            """C'est la vie. C'est la vie, vraiment!""",
            """Oui.
Non, je ne sais pas.""",
        ]
    )
    assert in_ids == [0, 0, 1, 1]
    assert p_ids == [0, 0, 0, 1]
    for i in sents:
        assert isinstance(i, str)


def test_sentence_join(qmt_translator):
    in_ids = [0, 0, 1, 1]
    p_ids = [0, 0, 0, 1]
    sents = ["C'est la vie.", "C'est la vie, vraiment!", "Oui.", "Non, je ne sais pas."]
    ret = qmt_translator._sentence_join(in_ids, p_ids, sents)
    ret == ["C'est la vie. C'est la vie, vraiment!", "Oui.\nNon, je ne sais pas."]
