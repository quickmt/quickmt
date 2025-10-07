import re
import sys

from fire import Fire
from itertools import islice, zip_longest

import numpy as np
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from torch import tensor
from torch.nn.functional import cosine_similarity
from tqdm import tqdm


def batch(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if not batch:
            return
        yield batch


def static_filter(
    src_dev: str,
    tgt_dev: str,
    src_input: str,
    src_output: str,
    tgt_input: str,
    tgt_output: str,
    src_bad_output: str,
    tgt_bad_output: str,
    batch_size: int = 4096,
    truncate_dimension: int = 256,
    limit: int = 100000,
    sim_cutoff_quantile: str = 0.02,
):
    model = SentenceTransformer(
        "sentence-transformers/static-similarity-mrl-multilingual-v1",
        device="cpu",
        truncate_dim=truncate_dimension,
    )
    # about 20k per second
    bad_indices = list()
    good_count = 0
    bad_count = 0
    keep_going = True

    with open(tgt_dev, "rt") as myfile:
        tgt_dev_txt = [i.strip() for i in myfile]

    with open(src_dev, "rt") as myfile:
        src_dev_txt = [i.strip() for i in myfile]

    # Find the distance for which 98% of true pairs in the dev set have similarity greater than this value
    # Then we will exclude pairs with similarity less than this value
    tgt_dev_embeddings = model.encode(
        tgt_dev_txt, batch_size=batch_size, convert_to_tensor=True
    )
    src_dev_embeddings = model.encode(
        src_dev_txt, batch_size=batch_size, convert_to_tensor=True
    )
    sims = cosine_similarity(tgt_dev_embeddings, src_dev_embeddings).numpy()
    cutoff = np.quantile(sims, sim_cutoff_quantile)
    print(f"Similarity cutoff: {cutoff}")

    with open(src_input, "rt") as src:
        with open(src_output, "wt") as src_out:
            with open(tgt_input, "rt") as tgt:
                with open(tgt_output, "wt") as tgt_out:
                    with open(src_bad_output, "wt") as src_bad_out:
                        with open(tgt_bad_output, "wt") as tgt_bad_out:
                            for x in tqdm(batch(zip(src, tgt), batch_size)):
                                tgt_embeddings = model.encode(
                                    [i[0].strip() for i in x],
                                    batch_size=batch_size,
                                    convert_to_tensor=True,
                                )
                                src_embeddings = model.encode(
                                    [i[1].strip() for i in x],
                                    batch_size=batch_size,
                                    convert_to_tensor=True,
                                )

                                sims = cosine_similarity(
                                    tgt_embeddings, src_embeddings
                                ).numpy()

                                for i, score in zip(x, sims):
                                    if score >= cutoff:
                                        src_out.write(i[0])
                                        tgt_out.write(i[1])
                                        good_count += 1
                                    else:
                                        src_bad_out.write(i[0])
                                        tgt_bad_out.write(i[1])
                                        bad_count += 1
    print(f"Good line count: {good_count}")
    print(f"Bad line count: {bad_count}")

def main():
    Fire(static_filter)

if __name__ == "__main__":
    main()
