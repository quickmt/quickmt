# `quickmt-zh-en` Neural Machine Translation Library 

A reasonably quick and reasonably accurate neural machine translation toolkit. Models are trained using [`eole`](https://github.com/eole-nlp/eole) and inference using [`ctranslate2`](https://github.com/OpenNMT/CTranslate2) with [`sentencepiece`](https://github.com/google/sentencepiece) for tokenization.

## Install `quickmt`

```bash
git clone https://github.com/quickmt/quickmt.git
pip install ./quickmt/
```

## Download model

```bash
# List available models
quickmt-list

quickmt-model-download quickmt/quickmt-zh-en ./quickmt-zh-en
```

## Use model

Inference with `quickmt`:

```python
from quickmt import Translator

# Auto-detects GPU, set to "cpu" to force CPU inference
t = Translator("./quickmt-zh-en/", device="auto")

# Translate - set beam size to 5 for higher quality (but slower speed)
t(["他补充道：“我们现在有 4 个月大没有糖尿病的老鼠，但它们曾经得过该病。”"], beam_size=1)

# Get alternative translations by sampling
# You can pass any cTranslate2 `translate_batch` arguments
t(["他补充道：“我们现在有 4 个月大没有糖尿病的老鼠，但它们曾经得过该病。”"], sampling_temperature=1.2, beam_size=1, sampling_topk=50, sampling_topp=0.9)
```

The model is in `ctranslate2` format, and the tokenizers are `sentencepiece`, so you can use the model files directly if you want. It would be fairly easy to get them to work with e.g. [LibreTranslate](https://libretranslate.com/) which also uses `ctranslate2` and `sentencepiece`.
