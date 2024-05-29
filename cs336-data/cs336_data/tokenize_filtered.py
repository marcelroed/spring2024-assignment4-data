from tqdm import tqdm
from pathlib import Path
from tiktoken import encoding_for_model
import numpy as np


enc = encoding_for_model('gpt2')
def tokenize_text(text: str):
    lines = text.splitlines()
    enc_lines = enc.encode_batch(lines, disallowed_special=())
    return enc_lines


def run_tokenize(in_dir: Path, out_path: Path):
    in_paths = list(in_dir.glob('*.txt'))
    all_tokens = []
    for in_path in tqdm(in_paths):
        with open(in_path, 'r') as f:
            text = f.read()
        tokens = tokenize_text(text)
        if len(all_tokens) != 0:
            all_tokens.append(50256)  # <|endoftext|>
        all_tokens.extend(tokens)
        # out_path = out_path / in_path.name
    nparr = np.array(all_tokens, dtype=np.uint16)
    nparr.tofile(out_path)

if __name__ == '__main__':
    run_tokenize(in_dir=Path('data/warc-train/'), out_path=Path('data/tokenized-train.np'))