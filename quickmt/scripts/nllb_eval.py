from time import time

import ctranslate2
import datasets
from fire import Fire
from sacrebleu import BLEU, CHRF, TER
from transformers import AutoTokenizer

bleu = BLEU()
chrf = CHRF()
ter = TER()


def main(
    output_file: str,
    src_lang_flores: str,
    tgt_lang_flores: str,
    ct2_model_path: str,
    device: str = "cuda",
    beam_size: int = 5,
    max_batch_size: int = 32,
    max_decoding_length: int = 512,
):
    try:
        flores = datasets.load_dataset(
            "facebook/flores",
            f"{src_lang_flores}-{tgt_lang_flores}",
        )
    except:
        flores = datasets.load_dataset(
            "facebook/flores",
            f"{tgt_lang_flores}-{src_lang_flores}",
        )

    src = []
    ref = []
    for i in flores["devtest"]:
        src.append(i[f"sentence_{src_lang_flores}"])
        ref.append(i[f"sentence_{tgt_lang_flores}"])

    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang=src_lang_flores)

    translator = ctranslate2.Translator(
        ct2_model_path,
        device=device,
        inter_threads=1,
        intra_threads=1,
    )

    # Tokenize
    t1 = time()
    input_tokens = [tokenizer.convert_ids_to_tokens(tokenizer.encode(i)) for i in src]

    # Translate
    results = translator.translate_batch(
        input_tokens,
        beam_size=beam_size,
        patience=1,
        max_decoding_length=max_decoding_length,
        max_batch_size=max_batch_size,
        target_prefix=[[tgt_lang_flores]] * len(input_tokens),
    )
    output_tokens = [i.hypotheses[0] for i in results]

    # Detokenize
    mt = [tokenizer.decode(tokenizer.convert_tokens_to_ids(i), skip_special_tokens=True) for i in output_tokens]
    mt = [i.replace("\n", "\t") for i in mt]

    t2 = time()

    print("Source sample: ", src[:10])
    print("Reference sample: ", ref[:10])
    print("Translation sample: ", mt[:10])

    print(f"Translation time: {t2-t1}")

    # Write results to file, for COMET scoring
    with open(output_file, "wt") as myfile:
        myfile.write("".join([i.replace("\n", "\t") + "\n" for i in mt]))

    # Print sacrebleu metrics
    print(bleu.corpus_score(mt, [ref]))
    print(chrf.corpus_score(mt, [ref]))
    print(ter.corpus_score(mt, [ref]))


if __name__ == "__main__":
    Fire(main)
