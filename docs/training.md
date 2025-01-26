
## Training `quickmt` Models

### Environment setup

```bash
# Install system dependencies
sudo apt install  libhunspell-dev parallel

## Install eole
git clone https://github.com/eole-nlp/eole.git
pip install -e ./eole

## Install ctranslate2
git clone --recursive https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2
mkdir build && cd build
cmake -DOPENMP_RUNTIME=COMP -DWITH_MKL=OFF .. 
make -j8
sudo make install
sudo ldconfig
pip install -e ./python/

# Install kenlm
pip install --config-settings="--build-option=--max_order=7" https://github.com/kpu/kenlm/archive/master.zip

# Install quickmt
python -m pip install -e ./
```

### Download Data

```bash
mv $HOME/.mtdata /path/to/large/disk
ln -s /path/to/large/disk $HOME/.mtdata

# Create experiment data/experiment folder
mkdir so-en

# List corpora
mtdata list -l som-eng | cut -f1

# Download corpora
mtdata get -l som-eng --out ./so-en --merge --no-fail  -j 2 \
--dev Flores-flores200_dev-1-eng-som OPUS-tatoeba-v20230412-eng-som \
--test Flores-flores200_devtest-1-eng-som  OPUS-ubuntu-v14.10-eng_CA-som \
--train Statmt-ccaligned-1-eng-som_SO ParaCrawl-paracrawl-1_bonus-eng-som  OPUS-ccaligned-v1-eng-som OPUS-ccmatrix-v1-eng-som  OPUS-multiccaligned-v1-eng-som OPUS-multiparacrawl-v7.1-eng-som OPUS-paracrawl-v7.1-eng-som OPUS-paracrawl-v8-eng-som OPUS-paracrawl-v9-eng-som OPUS-qed-v2.0a-eng-som OPUS-ted2020-v1-eng-som OPUS-tanzil-v1-eng-som OPUS-tatoeba-v20200531-eng-som OPUS-tatoeba-v20201109-eng-som OPUS-tatoeba-v20210310-eng-som OPUS-tatoeba-v20210722-eng-som OPUS-tatoeba-v20220303-eng-som OPUS-bible_uedin-v1-eng-som OPUS-infopankki-v1-eng-som OPUS-tico_19-v20201028-eng-som AllenAi-nllb-1-eng-som OPUS-gnome-v1-eng-som OPUS-ubuntu-v14.10-eng-som OPUS-xlent-v1.2-eng-som OPUS-wikimedia-v20230407-eng-som

# Move files to standardized src/tgt names 
cd so-en
mv dev.som dev.src
mv dev.eng dev.tgt
mv train.som train.src
mv train.eng train.tgt
```

### Clean Data a Bit

```{bash}
# Deduplicate 
# Will use /tmp, so make sure your disk is bit if your corpus is big
# Or set the TMPDIR env var to somewhere with a lot of space
paste -d '\t' train.src train.tgt \
    | sort | uniq  \
    | awk -v FS="\t" '{ print $1 > "train.uniq.src" ; print $2 > "train.uniq.tgt" }'

# Clean
paste -d '\t' train.uniq.src train.uniq.tgt | parallel -j 6 --pipe -k -l 100000  python -m quickmt.clean --src_lang so --tgt_lang en --src_min_langid_score 0 --tgt_min_langid_score 0.5 \
    | awk -v FS="\t" '{ print $1 > "train.clean.src" ; print $2 > "train.clean.tgt" }'

# Fix
time paste -d '\t' train.clean.src train.clean.tgt \
    | parallel -j 6 --pipe -k -l 100000 bifixer --scol 1 --tcol 2 --ignore_duplicates --ignore_segmentation  --tmp_dir  /media/mark/nvme2022/tmp -q - - so en  \
    | awk -v FS="\t" '{ print $1 > "train.fixed.src" ; print $2 > "train.fixed.tgt" }'

# Shuffle - https://www.baeldung.com/linux/randomize-file-lines
paste -d '\t' train.fixed.src train.fixed.tgt | \
    awk 'BEGIN{srand()}{print rand(), $0}' \
    | sort -n -k 1 \
    | awk 'sub(/\S* /,"")' \
    | awk -v FS="\t" '{ print $1 > "train.shuf.src" ; print $2 > "train.shuf.tgt" }'
```

### Upload Data to Huggingface

You will have to have authenticated to huggingface and you will need to write to a location for which you have permissions (replace `quickmt/quickmt-train.so-en` with `your_username/your_dataset_name`)

```
python -m quickmt.corpus_to_hf quickmt/quickmt-train.so-en --src_in train.shuf.src --tgt_in train.shuf.tgt --src_lang so --tgt_lang en
python -m quickmt.corpus_to_hf quickmt/quickmt-valid.so-en --src_in dev.src --tgt_in dev.tgt --src_lang so --tgt_lang en
```

### Train Tokenizers

```bash
# Train target tokenizer
spm_train --input_sentence_size 5000000 --shuffle_input_sentence false --input=train.shuf.tgt --num_threads 12 --model_prefix=tgt.spm --vocab_size=20000 --character_coverage=0.9995 --model_type=unigram

# Train source tokenizer
spm_train --input_sentence_size 5000000 --shuffle_input_sentence false --input=train.shuf.src --num_threads 12 --model_prefix=src.spm --vocab_size=20000 --character_coverage=0.9995 --model_type=unigram

# Convert spm vocab to eole vocab
cat tgt.spm.vocab | eole tools spm_to_vocab > tgt.eole.vocab
cat src.spm.vocab | eole tools spm_to_vocab > src.eole.vocab
```

### Train Model

```bash
eole train --config eole-config-small.yaml
```


### Inference with eole

```bash
eole predict -model_path ./so-en/model/ -src input.txt  -output output.txt  --batch_size 16 --gpu_ranks 0
```
 

### Convert to ctranslate2

```python
python -m ctranslate2.converters.eole_ct2 --model_path ./model --output_dir ./ct2-soen

# Copy over src and tgt tokenizers
cp src.spm.model ct2-soen/
cp tgt.spm.model ct2-soen/
```

### Evaluate

Evaluates on the `flores-devtest` dataset

```bash
python -m quickmt.eval --model_path ct2-soen --src_lang som_Latn --tgt_lang eng_Latn
```


