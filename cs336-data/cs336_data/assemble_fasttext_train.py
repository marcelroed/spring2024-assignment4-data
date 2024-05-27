from itertools import islice
from typing import Iterable
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from random import shuffle
import re
import gzip
from cs336_data.extract_text import iterwarc, extract_text_from_html_bytes, iterwarc_text

def ceildiv(a, b):
    return -(-a // b)


MULTIPLE_WHITESPACE = re.compile(r'\s+')

def load_jsonlines(path: str | Path) -> list[str]:
    jsonObj = pd.read_json(path_or_buf=path, lines=True)
    return jsonObj['text']

def load_all_jsonlines(dir: str | Path) -> Iterable[str]:
    for path in Path(dir).rglob('*.jsonl.gz'):
        yield from map(lambda s: re.sub(MULTIPLE_WHITESPACE, ' ', s), load_jsonlines(path))

def load_random_warc_entries(warc_dir: str | Path, n_entries: int):
    warc_files = list(Path(warc_dir).rglob('*.warc.filtered.gz'))
    entries_per_file = ceildiv(n_entries, len(warc_files))

    for warc_file in tqdm(warc_files, desc='Loading random WARC entries'):
        yield from islice(filter(lambda s: len(s) > 10, map(lambda s: re.sub(MULTIPLE_WHITESPACE, ' ', s), iterwarc_text(warc_file))), entries_per_file)

def prepend_label(text: str, label: str):
    return f'__label__{label} {text}'


if __name__ == '__main__':
    lines = list(map(lambda s: prepend_label(s, 'paloma'), load_all_jsonlines('data/paloma/')))
    warc_lines = list(map(lambda s: prepend_label(s, 'cc'), load_random_warc_entries('data/', n_entries=int(len(lines) * 1.3))))
    all_lines = lines + warc_lines
    shuffle(all_lines)
    with gzip.open('data/paloma-like-train.txt.gz', 'wt') as f:
        f.write('\n'.join(all_lines))

