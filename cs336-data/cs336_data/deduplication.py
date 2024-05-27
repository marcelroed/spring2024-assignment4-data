from collections import defaultdict
from pathlib import Path
import re
import string
import unicodedata
import networkx as nx
from shutil import copy as file_copy
import mmh3
from itertools import combinations

PUNCTUATION_RE = re.compile(f"([{re.escape(string.punctuation)}])")
WHITESPACE_RE = re.compile(r"\s+")

def exact_line_deduplication(input_paths: list[str | Path], out_path: str | Path):
    print(input_paths, out_path)
    seen_count = defaultdict(int)
    for input_path in input_paths:
        with open(input_path, 'r') as f:
            for line in f:
                line_hash = line.__hash__()
                seen_count[line_hash] += 1
    
    for input_path in input_paths:
        with open(input_path, 'r') as f, open(out_path / input_path.name, 'w') as out_f:
            for line in f:
                line_hash = line.__hash__()
                if seen_count[line_hash] == 1:
                        out_f.write(line)


def normalize_for_jaccard(s: str):
    s = s.lower()
    s = PUNCTUATION_RE.sub("", s)
    s = WHITESPACE_RE.sub(" ", s)

    # Normalize unicode characters if possible
    s = unicodedata.normalize("NFD", s)

    # Remove all accents that are in separate unicode codepoints
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")

    return s


def get_n_gram_set(s: str, num_words_n_gram: int) -> set[str]:
    """Set of all n-grams in the input string."""
    words = s.split()
    n_grams = set()
    for i in range(len(words) - num_words_n_gram + 1):
        n_gram = " ".join(words[i : i + num_words_n_gram])
        n_grams.add(n_gram)
    return n_grams


def min_hash_signature(set_of_n_grams: set, num_hashes: int) -> list[int]:
    min_hash_sig = []
    for i in range(num_hashes):
        min_hash = None
        for n_gram in set_of_n_grams:
            hash_ = mmh3.hash(n_gram, i)
            if min_hash is None or hash_ < min_hash:
                min_hash = hash_
        min_hash_sig.append(min_hash)
    return min_hash_sig


def jaccard_similarity(a: set, b: set):
    return len(a & b) / len(a | b)

def minhash_deduplication(in_paths: list[Path | str], num_hashes: int, num_bands: int, ngrams: int, out_dir: str | Path, jaccard_threshold: float | None = None):
    if jaccard_threshold is None:
        jaccard_threshold = (1 / num_bands) ** (1 / band_size)

    band_size = num_hashes // num_bands
    band_bucket = [defaultdict(list) for _ in range(num_bands)]
    for in_path in in_paths:
        with open(in_path, "r") as f:
            s = f.read()

        s = normalize_for_jaccard(s)
        n_gram_set = get_n_gram_set(s, ngrams)
        min_hash_sig = min_hash_signature(n_gram_set, num_hashes)

        for band_idx in range(num_bands):
            hash_beg_idx = band_idx * band_size
            hash_end_idx = (band_idx + 1) * band_size
            key = tuple(min_hash_sig[hash_beg_idx:hash_end_idx])
            val = in_path
            band_bucket[band_idx][key].append(val)

    to_add_pairs = set()
    for bucket_dict in band_bucket:
        for bucket in bucket_dict.values():
            if len(bucket) == 1:
                continue
            for pair in combinations(bucket, 2):
                to_add_pairs.add(pair)


    similar = []
    for pair in to_add_pairs:
        in_path1, in_path2 = pair
        with open(in_path1, "r") as f1:
            s1 = f1.read()
        with open(in_path2, "r") as f2:
            s2 = f2.read()
        n_gram_set1 = get_n_gram_set(s1, ngrams)
        n_gram_set2 = get_n_gram_set(s2, ngrams)
        true_similarity = jaccard_similarity(n_gram_set1, n_gram_set2)
        if true_similarity > jaccard_threshold:
            similar.append(pair)

    g = nx.Graph()
    g.add_nodes_from(in_paths)
    g.add_edges_from(similar)
    components: list[set[str]] = list(nx.connected_components(g))

    for component in components:
        in_path = component.pop()
        out_path = f"{out_dir}/{Path(in_path).name}"
        file_copy(in_path, out_path)
