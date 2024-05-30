from tqdm import tqdm
from pathlib import Path
from tiktoken import encoding_for_model
import numpy as np
import pandas as pd
import re

MULTIPLE_WHITESPACE = re.compile(r'\s+')


enc = encoding_for_model('gpt2')
def tokenize_text(text: str | list[str]):
    if not isinstance(text, list):
        lines = text.splitlines()
    else:
        lines = text
    enc_lines = enc.encode_batch(lines, disallowed_special=())
    return enc_lines


def run_tokenize(in_dir: Path, out_path: Path):
    in_paths = list(in_dir.glob('*.txt'))
    all_tokens = []
    for in_path in tqdm(in_paths):
        with open(in_path, 'r') as f:
            text = f.read()
        tokens_for_file = tokenize_text(text)
        for tokens in tokens_for_file:
            if len(all_tokens) != 0:
                all_tokens.append(50256)  # <|endoftext|>
            all_tokens.extend(tokens)
        # out_path = out_path / in_path.name
    nparr = np.array(all_tokens, dtype=np.uint16)
    nparr.tofile(out_path)

def tokenize_single(in_path: Path, out_path: Path):
    all_tokens = []
    with open(in_path, 'r') as f:
        text = f.read()
    tokens_for_file = tokenize_text(text)
    for tokens in tokens_for_file:
        if len(all_tokens) != 0:
            all_tokens.append(50256)  # <|endoftext|>
        all_tokens.extend(tokens)
    # out_path = out_path / in_path.name
    nparr = np.array(all_tokens, dtype=np.uint16)
    nparr.tofile(out_path)


def run_tokenize_parallel(in_dir: Path, out_dir: Path):
    import concurrent.futures
    import multiprocessing as mp
    import os
    num_cpus = len(os.sched_getaffinity(0))
    print(f'Running on {num_cpus} CPUs')
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus // 2, mp_context=mp.get_context('forkserver'))
    futures = []
    in_paths = list(in_dir.glob('*.txt'))
    for in_path in tqdm(in_paths):
        # tokenize_single(in_path, out_dir / in_path.name.replace('.txt', '.np'))
        # For each WARC filepath, submit a job to the executor and get a future back
        future = executor.submit(
            tokenize_single,
            in_path,
            out_dir / in_path.name.replace('.txt', '.np'),
        )
        # Store the futures
        futures.append(future)
    # Iterate over the completed futures as they finish, using a progress bar # to keep track of progress.
    for future in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(in_paths),
        smoothing=0.0,
        desc='Total WARC progress'
    ):
        output_file = future.result()
        # print(f"Output file written: {output_file}")


def merge_tokenized(in_dir: Path, out_path: Path):
    all_tokens = []
    for in_path in tqdm(list(in_dir.glob('*.np'))):
        tokens = np.fromfile(in_path, dtype=np.uint16)
        all_tokens.append(tokens)
    
    nparr = np.concatenate(all_tokens)
    
    nparr.tofile(out_path)



def run_tokenize_paloma(in_dir: Path, out_path: Path):
    all_tokens = []
    for in_path in tqdm(list(in_dir.glob('*.jsonl.gz'))):
        texts = pd.read_json(in_path, lines=True).text.tolist()
        pre_tokenize = []
        for text in texts:
            text = MULTIPLE_WHITESPACE.sub(' ', text)
            pre_tokenize.append(text)
        tokens_for_line = tokenize_text(pre_tokenize)
        for tokens in tokens_for_line:
            if len(all_tokens) != 0:
                all_tokens.append(50256)
            all_tokens.extend(tokens)
        # out_path = out_path / in_path.name
    print(all_tokens[:2])
    nparr = np.array(all_tokens, dtype=np.uint16)
    nparr.tofile(out_path)

if __name__ == '__main__':
    # run_tokenize_paloma(in_dir=Path('data/paloma/'), out_path=Path('data/tokenized-paloma.np'))
    # run_tokenize(in_dir=Path('data/warc-train/'), out_path=Path('data/tokenized-train.np'))
    # run_tokenize_parallel(in_dir=Path('data/warc-train/'), out_dir=Path('data/tokenized-train/'))
    merge_tokenized(in_dir=Path('data/tokenized-train/'), out_path=Path('data/tokenized-train.np'))