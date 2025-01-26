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
    install_requires=[
        #"eole>=0.1.0",
        "ctranslate2>=4,<5",
        "sentencepiece",
        "fasttext-wheel",
        "blingfire",
        "fire",
        "bicleaner-hardrules",
        "bifixer",
        "fasttext-wheel",
        "mtdata",
        "nltk"
    ]
)

