from abc import ABC, abstractmethod
from pathlib import Path
from time import time
from typing import List, Optional, Union

import ctranslate2
import sentencepiece
from blingfire import text_to_sentences
from pydantic import DirectoryPath, validate_call
from huggingface_hub import snapshot_download


class TranslatorABC(ABC):
    def __init__(self, model_path: DirectoryPath, **kwargs):
        """Create quickmt translation object

        Args:
            model_path (DirectoryPath): Path to quickmt model folder
            **kwargs: CTranslate2 Translator arguments - see https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html
        """
        self.model_path = Path(model_path)
        self.translator = ctranslate2.Translator(str(model_path), **kwargs)

    @staticmethod
    @validate_call
    def _sentence_split(src: List[str]):
        """Split sentences with Blingfire

        Args:
            src (List[str]): Input list of strings to split by sentences

        Returns:
            List[int], List[int], List[str]: List of input ids, list of paragraph ids and sentences
        """
        input_ids = []
        paragraph_ids = []
        sentences = []
        for idx, i in enumerate(src):
            for paragraph, j in enumerate(i.splitlines(keepends=True)):
                sents = text_to_sentences(j).splitlines()
                for sent in sents:
                    stripped_sent = sent.strip()
                    if len(stripped_sent) > 0:
                        if (
                            len(stripped_sent) < 5
                            and len(paragraph_ids) > 0
                            and paragraph == paragraph_ids[-1]
                            and len(input_ids) > 0
                            and input_ids[-1] == idx
                        ):
                            sentences[-1] += " " + stripped_sent
                        else:
                            input_ids.append(idx)
                            paragraph_ids.append(paragraph)
                            sentences.append(stripped_sent)

        return input_ids, paragraph_ids, sentences

    @staticmethod
    @validate_call
    def _sentence_join(
        input_ids: List[int],
        paragraph_ids: List[int],
        sentences: List[str],
        paragraph_join_str: str = "\n",
        sent_join_str: str = " ",
        length: Optional[int] = None,
    ):
        """Sentence joiner

        Args:
            input_ids (List[int]): List of input IDs
            paragraph_ids (List[int]): List of paragraph IDs
            sentences (List[str]): List of sentences to join up by input and paragraph ids
            paragraph_join_str (str, optional): str to use to join paragraphs. Defaults to "\n".
            sent_join_str (str, optional): str to join up sentences. Defaults to " ".

        Returns:
            List[str]: Joined up sentences
        """
        if not input_ids:
            return [""] * (length or 0)

        target_len = length if length is not None else (max(input_ids) + 1)
        ret = [""] * target_len
        last_paragraph = 0
        for idx, paragraph, text in zip(input_ids, paragraph_ids, sentences):
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

    @abstractmethod
    def tokenize(
        self,
        sentences: List[str],
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
    ): ...

    @abstractmethod
    def detokenize(
        self,
        sentences: List[List[str]],
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
    ): ...

    @abstractmethod
    def translate_batch(
        self,
        sentences: List[List[str]],
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
    ): ...

    @abstractmethod
    def unload(self): ...

    @validate_call
    def __call__(
        self,
        src: Union[str, List[str]],
        max_batch_size: int = 32,
        max_decoding_length: int = 256,
        beam_size: int = 2,
        patience: int = 1,
        length_penalty: float = 1.0,
        coverage_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        verbose: bool = False,
        src_lang: Union[None, str] = None,
        tgt_lang: Union[None, str] = None,
        **kwargs,
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

        indices, paragraphs, sentences = self._sentence_split(src)

        if not sentences:
            return "" if return_string else [""] * len(src)

        if verbose:
            print(f"Split sentences: {sentences}")

        input_text = self.tokenize(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
        if verbose:
            print(f"Tokenized input: {input_text}")

        t1 = time()
        results = self.translate_batch(
            input_text,
            beam_size=beam_size,
            patience=patience,
            length_penalty=length_penalty,
            coverage_penalty=coverage_penalty,
            repetition_penalty=repetition_penalty,
            max_decoding_length=max_decoding_length,
            max_batch_size=max_batch_size,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            **kwargs,
        )
        t2 = time()
        if verbose:
            print(f"Translation time: {t2 - t1}")

        output_tokens = [i.hypotheses[0] for i in results]

        if verbose:
            print(f"Tokenized output: {output_tokens}")

        translated_sents = self.detokenize(
            output_tokens, src_lang=src_lang, tgt_lang=tgt_lang
        )

        ret = self._sentence_join(
            indices, paragraphs, translated_sents, length=len(src)
        )

        if return_string:
            return ret[0]
        else:
            return ret

    @validate_call
    def translate_file(self, input_file: str, output_file: str, **kwargs) -> None:
        """Translate a file with a quickmt model

        Args:
            file_path (str): Path to plain-text file to translate
        """
        with open(input_file, "rt") as myfile:
            src = myfile.readlines()

        # Remove newlines
        src = [i.strip() for i in src]

        # Translate
        mt = self(src, **kwargs)

        # Replace newlines to ensure output is the same number of lines
        mt = [i.replace("\n", "\t") for i in mt]

        with open(output_file, "wt") as myfile:
            myfile.write("".join([i + "\n" for i in mt]))

    @validate_call
    def translate_stream(
        self,
        src: Union[str, List[str]],
        max_batch_size: int = 32,
        max_decoding_length: int = 256,
        beam_size: int = 5,
        patience: int = 1,
        src_lang: Union[None, str] = None,
        tgt_lang: Union[None, str] = None,
        **kwargs,
    ):
        """Translate a list of strings with quickmt model

        Args:
            src (List[str]): Input list of strings to translate
            max_batch_size (int, optional): Maximum batch size, to constrain RAM utilization. Defaults to 32.
            beam_size (int, optional): CTranslate2 Beam size. Defaults to 5.
            patience (int, optional): CTranslate2 Patience. Defaults to 1.
            max_decoding_length (int, optional): Maximum length of translation
            **args: Other CTranslate2 translate_batch args, see https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html#ctranslate2.Translator.translate_batch
        """
        if isinstance(src, str):
            src = [src]

        indices, paragraphs, sentences = self._sentence_split(src)

        input_text = self.tokenize(sentences, src_lang=src_lang, tgt_lang=tgt_lang)

        translations_iterator = self.translator.translate_iterable(
            input_text,
            beam_size=beam_size,
            patience=patience,
            max_decoding_length=max_decoding_length,
            max_batch_size=max_batch_size,
            **kwargs,
        )

        for idx, para, sent, output in zip(
            indices, paragraphs, sentences, translations_iterator
        ):
            yield {
                "input_idx": idx,
                "sentence_idx": para,
                "input_text": sent,
                "translation": self.detokenize([output.hypotheses[0]])[0],
            }

    def translate(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)


class Translator(TranslatorABC):
    def __init__(
        self,
        model_path: str | DirectoryPath,
        inter_threads: int = 1,
        intra_threads: int = 0,
        **kwargs,
    ):
        """Create quickmt translation object

        Args:
            model_path (str | DirectoryPath): Quickmt Model ID or path to quickmt model folder
            inter_threads (int): Number of simultaneous translations
            intra_threads (int): Number of threads for each translation
            **kwargs: CTranslate2 Translator arguments - see https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html
        """
        # snapshot_download returns the local path in the HF cache.
        # Try local only first to speed up loading
        if Path(model_path).exists():
            model_folder = model_path
        else:
            try:
                model_folder = snapshot_download(
                    repo_id=model_path,
                    ignore_patterns=["eole-model/*", "eole_model/*"],
                    local_files_only=True,
                )
            except Exception:
                model_folder = snapshot_download(
                    repo_id=model_path,
                    ignore_patterns=["eole-model/*", "eole_model/*"],
                )

        super().__init__(
            model_folder,
            inter_threads=inter_threads,
            intra_threads=intra_threads,
            **kwargs,
        )
        joint_tokenizer_path = self.model_path / "joint.spm.model"
        if joint_tokenizer_path.exists():
            self.source_tokenizer = sentencepiece.SentencePieceProcessor(
                model_file=str(self.model_path / "joint.spm.model")
            )
            self.target_tokenizer = sentencepiece.SentencePieceProcessor(
                model_file=str(self.model_path / "joint.spm.model")
            )
        else:
            self.source_tokenizer = sentencepiece.SentencePieceProcessor(
                model_file=str(self.model_path / "src.spm.model")
            )
            self.target_tokenizer = sentencepiece.SentencePieceProcessor(
                model_file=str(self.model_path / "tgt.spm.model")
            )

    def __del__(self):
        self.unload()

    def tokenize(
        self,
        sentences: List[str],
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
    ):
        # Default implementation ignores lang tags unless explicitly handled
        return [
            i + ["</s>"] for i in self.source_tokenizer.encode(sentences, out_type=str)
        ]

    def detokenize(
        self,
        sentences: List[List[str]],
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
    ):
        return self.target_tokenizer.decode(sentences)

    def unload(self):
        """Explicitly release CTranslate2 translator resources"""
        if hasattr(self, "translator"):
            del self.translator

    def translate_batch(
        self,
        input_text: List[List[str]],
        beam_size: int = 5,
        patience: int = 1,
        max_decoding_length: int = 256,
        max_batch_size: int = 32,
        disable_unk: bool = True,
        replace_unknowns: bool = False,
        length_penalty: float = 1.0,
        coverage_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        src_lang: str = None,
        tgt_lang: str = None,
        **kwargs,
    ):
        """Translate a list of strings

        Args:
            input_text (List[List[str]]): Input text to be translated
            beam_size (int, optional): Beam size for beam search. Defaults to 5.
            patience (int, optional): Stop beam search when `patience` beams finish. Defaults to 1.
            max_decoding_length (int, optional): Max decoding length for model. Defaults to 256.
            max_batch_size (int, optional): Max batch size. Reduce to limit RAM usage. Increase for faster speed. Defaults to 32.
            disable_unk (bool, optional): Disable generating unk token. Defaults to True.
            replace_unknowns (bool, optional): Replace unk tokens with src token that has the highest attention value. Defaults to False.
            length_penalty (float, optional): Length penalty. Defaults to 1.0.
            coverage_penalty (float, optional): Coverage penalty. Defaults to 0.0.
            src_lang (str, optional): Source language. Only needed for multilingual models. Defaults to None.
            tgt_lang (str, optional): Target language. Only needed for multilingual models. Defaults to None.

        Returns:
            List[str]: Translated text
        """
        return self.translator.translate_batch(
            input_text,
            beam_size=beam_size,
            patience=patience,
            max_decoding_length=max_decoding_length,
            max_batch_size=max_batch_size,
            disable_unk=disable_unk,
            replace_unknowns=replace_unknowns,
            length_penalty=length_penalty,
            coverage_penalty=coverage_penalty,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )
