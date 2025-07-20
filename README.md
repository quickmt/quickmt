# `quickmt` Neural Machine Translation Library 

A reasonably quick and reasonably accurate neural machine translation toolkit. Models are trained using [`eole`](https://github.com/eole-nlp/eole) and inference using [`ctranslate2`](https://github.com/OpenNMT/CTranslate2) with [`sentencepiece`](https://github.com/google/sentencepiece) for tokenization.

## Why `quickmt`?

Ten out of the top twenty most downloaded machine translation (MT) models on Huggingface are `Helsinki-NLP/opus-mt-xx-xx` models. The French to English MT model was downloaded 820,000+ times in the past month. It is common to download pre-trained models from Huggingface and then fine-tune them to be better for specific tasks, but surely the majority of these downloads are people intending to *use these models as-is* rather than fine-tune them. 

![Top MT Models on Huggingface](docs/blogs/img/top-hf-translation-models.png)

The `quickmt` project was created to provide alternative translation models for high-resource languages that are faster *and* more accurate than the `opus-mt` series of models. I'm not aiming for world-class accuracy - if you have enough compute (or money to use a hosted service) you will be better off using a high-quality general-purpose LLM like [`meta-llama/Llama-3.3-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) or an LLM fine-tune for MT like [`Unbabel/Tower-Plus-72B`](https://huggingface.co/Unbabel/Tower-Plus-72B). I am aiming for a *reasonably* high-quality and *relatively* fast alternative to the very popular `opus-mt` models. 

`quickmt` models are 3x faster than `opus-mt` models:

![opus-mt vs. quickmt speed comparison](docs/blogs/img/quickmt-opusmt-speed.png)

*And* higher quality:

![opus-mt vs. quickmt quality comparison](docs/blogs/img/quickmt-vs-opusmt-to-english.png)

![opus-mt vs. quickmt quality comparison](docs/blogs/img/quickmt-vs-opusmt-from-english.png)

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
