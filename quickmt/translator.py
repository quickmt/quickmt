from pathlib import Path
from time import time
from typing import List, Tuple

import ctranslate2
import sentencepiece
from blingfire import text_to_sentences
from pydantic import DirectoryPath, validate_call


class Translator:
    def __init__(self, model_path: DirectoryPath, **args):
        self.model_path = Path(model_path)
        self.translator = ctranslate2.Translator(model_path, **args)
        self.source_tokenizer = sentencepiece.SentencePieceProcessor(str(self.model_path / "src.spm.model"))
        self.target_tokenizer = sentencepiece.SentencePieceProcessor(str(self.model_path / "tgt.spm.model"))

    @staticmethod
    @validate_call
    def sentence_split(src: List[str]):
        """Split sentences with Blingfire

        Args:
            src (List[str]): Input list of strings to split by sentences

        Returns:
            List[Tuple[int, str]]: List of tuples, first element is the input index, second element is the sentence
        """
        ret = []
        for idx, i in enumerate(src):
            for j in i.splitlines(keepends=True):
                sents = text_to_sentences(j).splitlines()
                for sent in sents:
                    stripped_sent = sent.strip()
                    if len(stripped_sent)>0:
                        # If the next segment is just a few chars, just tack it on
                        if len(stripped_sent) < 5:
                            ret[-1][1] += stripped_sent
                        else:
                            ret.append([idx, stripped_sent])
        return ret

    @staticmethod
    @validate_call
    def sentence_join(src: List[Tuple[int, str]], sent_join_str: str = " "):
        """Join sentences together

        Args:
            src (List[int, str]): Input list of indices and strings to join back up

        Returns:
            List[str]: List of strings, joined by join key
        """
        ret = ["" for _ in range(1 + max([i[0] for i in src]))]
        for idx, i in src:
            ret[idx] += sent_join_str + i
        return [i.strip() for i in ret]

    @validate_call
    def __call__(self, src: List[str], max_batch_size: int = 32, beam_size: int = 5, patience: int = 1):
        """Translate a list of strings with quickmt model

        Args:
            src (List[str]): Input list of strings to translate
            max_batch_size (int, optional): Maximum batch size, to constrain RAM utilization. Defaults to 32.
            beam_size (int, optional): CTranslate2 Beam size. Defaults to 5.
            patience (int, optional): CTranslate2 Patience. Defaults to 1.

        Returns:
            List[str]: Translation of the input
        """
        sentences = self.sentence_split(src)

        input_tokens = self.source_tokenizer.encode([i[1] for i in sentences], out_type=str)

        t1 = time()
        results = self.translator.translate_batch(
            input_tokens,
            beam_size=beam_size,
            patience=patience,
            max_decoding_length=512,
            max_batch_size=max_batch_size,
        )
        t2 = time()
        print(f"Translation time: {t2-t1}")

        output_tokens = [i.hypotheses[0] for i in results]

        translated_sents = [self.target_tokenizer.decode(i) for i in output_tokens]

        return self.sentence_join(
            (idx, translation) for idx, translation in zip([i[0] for i in sentences], translated_sents)
        )
