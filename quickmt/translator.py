from pathlib import Path
from time import time
from typing import List, Tuple, Union

import ctranslate2
import sentencepiece
from blingfire import text_to_sentences
from pydantic import DirectoryPath, validate_call


class Translator:
    def __init__(self, model_path: DirectoryPath, **args):
        """Create quickmt translation object

        Args:
            model_path (DirectoryPath): Path to quickmt model folder
            **args: CTranslate2 Translator arguments - see https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html
        """
        self.model_path = Path(model_path)
        self.translator = ctranslate2.Translator(model_path, **args)
        self.source_tokenizer = sentencepiece.SentencePieceProcessor(str(self.model_path / "src.spm.model"))
        self.target_tokenizer = sentencepiece.SentencePieceProcessor(str(self.model_path / "tgt.spm.model"))

    @staticmethod
    @validate_call
    def _sentence_split(src: List[str]):
        """Split sentences with Blingfire

        Args:
            src (List[str]): Input list of strings to split by sentences

        Returns:
            List[List[int, str]]: List of tuples, first element is the input index, second element is the sentence
        """
        ret = []
        for idx, i in enumerate(src):
            for paragraph, j in enumerate(i.splitlines(keepends=True)):
                sents = text_to_sentences(j).splitlines()
                for sent in sents:
                    stripped_sent = sent.strip()
                    if len(stripped_sent) > 0:
                        # If the next segment is just a few chars, just tack it on
                        if (len(ret) > 0 and idx == ret[-1][0]) and (len(stripped_sent) < 5):
                            ret[-1][2] += stripped_sent
                        else:
                            ret.append([idx, paragraph, stripped_sent])
        return ret

    @staticmethod
    @validate_call
    def _sentence_join(src: List[Tuple[int, int, str]], paragraph_join_str: str = "\n", sent_join_str: str = " "):
        """Join sentences together

        Args:
            src (List[int, int,  str]): Input list of indices and strings to join back up

        Returns:
            List[str]: List of strings, joined by join key
        """
        ret = list(["" for _ in range(1 + max([i[0] for i in src]))])
        last_paragraph = 0
        for idx, paragraph, text in src:
            if len(ret[idx]) > 0:
                if paragraph == last_paragraph:
                    ret[idx] += sent_join_str + text
                else:
                    ret[idx] += paragraph_join_str + text
                last_paragraph = paragraph
            else:
                ret[idx] = text
                last_paragraph = paragraph
        return ret

    @validate_call
    def __call__(
        self,
        src: Union[str, List[str]],
        max_batch_size: int = 32,
        max_decoding_length: int = 512,
        beam_size: int = 4,
        patience: int = 1,
        **args,
    ) -> Union[str, List[str]]:
        """Translate a list of strings with quickmt model

        Args:
            src (List[str]): Input list of strings to translate
            max_batch_size (int, optional): Maximum batch size, to constrain RAM utilization. Defaults to 32.
            beam_size (int, optional): CTranslate2 Beam size. Defaults to 5.
            patience (int, optional): CTranslate2 Patience. Defaults to 1.
            max_decoding_length (int, optional): Maximum length of translation
            **args: Other CTranslate2 translate_batch args, see https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html#ctranslate2.Translator.translate_batch

        Returns:
            Union[str, List[str]]: Translation of the input
        """
        if isinstance(src, str):
            return_string = True
            src = [src]
        else:
            return_string = False

        sentences = self._sentence_split(src)

        input_tokens = self.source_tokenizer.encode([i[2] for i in sentences], out_type=str)

        t1 = time()
        results = self.translator.translate_batch(
            input_tokens,
            beam_size=beam_size,
            patience=patience,
            max_decoding_length=max_decoding_length,
            max_batch_size=max_batch_size,
            **args,
        )
        t2 = time()
        print(f"Translation time: {t2-t1}")

        output_tokens = [i.hypotheses[0] for i in results]

        translated_sents = [self.target_tokenizer.decode(i) for i in output_tokens]

        indices = [i[0] for i in sentences]
        paragraphs = [i[1] for i in sentences]

        ret = self._sentence_join(
            (
                (idx, paragraph, translation)
                for idx, paragraph, translation in zip(indices, paragraphs, translated_sents)
            )
        )

        if return_string:
            return ret[0]
        else:
            return ret
