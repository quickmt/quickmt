from pathlib import Path
from time import time

import ctranslate2
import datasets
import sentencepiece as spm
from fire import Fire
from sacrebleu import BLEU, CHRF

bleu = BLEU()
chrf = CHRF()


def eval(model_path: str, src_lang: str, tgt_lang: str, num_threads: int = 6, compute_type="auto", device: str = "cpu"):
    model_path = Path(model_path)

    print("Loading flores-devtest data")
    try:
        flores = datasets.load_dataset(
            "facebook/flores",
            f"{src_lang}-{tgt_lang}",  # trust_remote_code=True
        )
    except:
        flores = datasets.load_dataset(
            "facebook/flores",
            f"{tgt_lang}-{src_lang}",  # trust_remote_code=True
        )

    src = []
    ref = []
    for i in flores["devtest"]:
        src.append(i[f"sentence_{src_lang}"])
        ref.append(i[f"sentence_{tgt_lang}"])

    if device == "cpu":
        device = "cpu"
        num_threads = num_threads
    else:
        device = "cuda"
        num_threads = 1

    print("Loading translation model")
    translator = ctranslate2.Translator(
        str(model_path),
        device=device,
        compute_type=compute_type,
        inter_threads=num_threads,
        intra_threads=1,
    )
    sp = spm.SentencePieceProcessor(str(model_path / "src.spm.model"))
    sp2 = spm.SentencePieceProcessor(str(model_path / "tgt.spm.model"))

    print("Tokenizing data")
    input_tokens = sp.encode(src, out_type=str)

    print("Translating")
    t1 = time()
    results = translator.translate_batch(
        input_tokens,
        beam_size=5,
        patience=1,
        max_decoding_length=512,
        max_batch_size=32,
    )
    t2 = time()
    print(f"Translation time: {t2-t1}")

    output_tokens = [i.hypotheses[0] for i in results]
    mt = [sp2.decode(i) for i in output_tokens]
    print(mt[:5])

    print(bleu.corpus_score(mt, [ref]))
    print(chrf.corpus_score(mt, [ref]))


def main():
    Fire(eval)


if __name__ == "__main__":
    main()
