from pathlib import Path

import numpy as np
from fasttext import load_model
from cs336_data.extract_text import iterwarc, extract_text_from_html_bytes

class IdentifyLanguageModel:
    def __init__(self, path='data/lid.176.bin'):
        self.model = load_model(path)

    def predict(self, text):
        label, prob = self.model.predict(text.replace('\n', ' '))
        print(label, prob)
        label = label[0].replace('__label__', '')
        return label, prob[0]


def _get_indices(iterator, sorted_idcs: list[int]):
    for i, entry in enumerate(iterator):
        if i == sorted_idcs[0]:
            yield entry
            sorted_idcs = sorted_idcs[1:]
            if len(sorted_idcs) == 0:
                break

def _get_fixed_random_warc_records(warc_path: str | Path, n_entries: int):
    np.random.seed(0)
    # Pick from the first 10_000 entries
    idcs = np.random.permutation(10_000)[:n_entries]
    idcs = np.sort(idcs)
    entries = iterwarc(warc_path)
    yield from _get_indices(entries, idcs)

def _run_language_identification():
    random_records = _get_fixed_random_warc_records('data/CC-MAIN-20180420081400-20180420101400-00118.warc.gz', n_entries=20)
    identifier_model = IdentifyLanguageModel()
    for record in random_records:
        text = extract_text_from_html_bytes(record.reader.read())
        print(text)
        print(identifier_model.predict(text))
        input()


def main():
    _run_language_identification()


if __name__ == '__main__':
    main()