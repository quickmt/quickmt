#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="quickmt",
    description="Relatively Quick Neural Machine Translation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.0.1",
    packages=find_packages(),
    project_urls={
        "Source": "https://github.com/quickmt/quickmt",
    },
    python_requires=">=3.10",
    install_requires=["ctranslate2>=4,<5", "sentencepiece", "blingfire", "fire", "pydantic>2", "huggingface_hub", "datasets", "sacrebleu"],
    extras_require={"dev": ["eole>=0.1.0", "bifixer", "fasttext-wheel", "mtdata", "nltk"]},
    entry_points={
        "console_scripts": [
            "quickmt-clean=quickmt.scripts.clean:main",
            "quickmt-eval=quickmt.scripts.eval:main",
            "quickmt-list=quickmt.hub:list",
            "quickmt-model-download=quickmt.hub:download",
            "quickmt-model-upload=quickmt.hub:upload",
            "quickmt-corpus-upload=quickmt.scripts.corpus_to_hf:main",
        ],
    },
)
