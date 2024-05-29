import fasttext
fasttext.FastText.eprint = lambda x: None
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from itertools import islice
from typing import BinaryIO
from tqdm import tqdm
import re
import fsspec
import multiprocessing as mp
import s3fs

# s3 = fsspec.filesystem('s3')

from fasttext import load_model

from cs336_data.deduplication import minhash_deduplication
from cs336_data.extract_text import extract_text_from_html_bytes, iterwarc_text
from cs336_data.harmful_content import NSFWModel, ToxicSpeechModel
from cs336_data.identify_language import IdentifyLanguageModel
from cs336_data.quality_classifier import QualityModel, train_fasttext_model
from cs336_data.quality_filter import gopher_quality_filter, GopherModel

MULTIPLE_WHITESPACE = re.compile(r'\s+')
    

class PalomaLikeModel:
    def __init__(self, model_path: str | Path = 'data/paloma-like.bin'):
        self.model = load_model(str(model_path))
    
    def predict(self, text: str):
        label, prob = self.model.predict(text.replace('\n', ' '))
        label = label[0].replace('__label__', '')
        return label, prob[0]
        

def load_and_predict(text: str):
    model = QualityModel('data/fasttext-quality.bin')
    return model.predict(text)




@dataclass
class Stats:
    retained: int = 0
    removed: int = 0

    def kept_ratio(self):
        return self.retained / self.total_seen()
    
    def total_seen(self):
        return self.retained + self.removed

class Filterer:
    def __init__(self, include_paloma=False):
        self.language_model = IdentifyLanguageModel()
        self.nsfw_model = NSFWModel()
        self.toxic_model = ToxicSpeechModel()
        self.paloma_like_model = PalomaLikeModel()
        # self.quality_model = QualityModel()
        self.gopher_model = GopherModel()
        self.stats = defaultdict(Stats)
        self.model_args_list = [
            (self.language_model, 'en', 0.5), (self.gopher_model, 'keep', 0.5), (self.nsfw_model, 'non-nsfw', 0.7), (self.toxic_model, 'non-toxic', 0.8)
        ]
        if include_paloma:
            self.model_args_list.append((self.paloma_like_model, 'paloma', 0.00000000001))
    
    def __call__(self, text):
        for model_args in self.model_args_list:
            if isinstance(model_args[0], PalomaLikeModel):
                text = MULTIPLE_WHITESPACE.sub(' ', text)
            model_keep = self.use_model(*model_args, text)
            if not model_keep:
                return False
        return True
    
    def filter_iterable(self, it):
        for text in it:
            if self(text):
                yield text
    
    def use_model(self, model, keep_label, keep_threshold, text):
        stats = self.stats[model.__class__.__name__]
        label, prob = model.predict(text)
        # if isinstance(model, PalomaLikeModel):
        #     print(label, prob)
        if isinstance(model, PalomaLikeModel) and label == 'cc':
            label = 'paloma'
            prob = 1 - prob
        if label == keep_label and prob >= keep_threshold:
            stats.retained += 1
            return True
        else:
            stats.removed += 1
            return False
        
    def show_stats(self) -> str:
        out_str = []
        for k, v in self.stats.items():
            out_str.append(f'{k}: {v.retained} retained, {v.removed} removed ({v.kept_ratio() * 100:.2f}% kept)')
        return '\n'.join(out_str)


def run_filter(warc_path: str | Path | BinaryIO, limit=None, id: int = 0):
    filterer = Filterer(include_paloma=True)
    if id == 0:
        it = tqdm(iterwarc_text(warc_path), smoothing=0.1, desc='Filtering WARC')
    else:
        it = iterwarc_text(warc_path)

    for text in islice(it, limit):
        if filterer(text):
            yield MULTIPLE_WHITESPACE.sub(' ', text)
    print(f'WARC: {warc_path}\n{filterer.show_stats()}')

def process_single_warc_file(id: int, input_path: Path, output_path: Path):
    try:
        # TODO: read input path, process the input, and write the output to output_path
        s3 = s3fs.S3FileSystem()
        # texts = run_filter(warc_path=input_path, id=id)
        f = s3.open(f'ai-residency-stanford-snap-uce/dataset/mds/CC/{input_path.name}', 'rb', block_size=2**25)
        texts = run_filter(warc_path=f, id=id)
        texts = '\n'.join(texts)
        with open(output_path, 'w') as f:
            f.write(texts)
        return output_path
    except:
        pass

def parallel_run_filter():
    import concurrent.futures
    import os
    # Set up the executor
    num_cpus = len(os.sched_getaffinity(0))
    print(f'Running on {num_cpus} CPUs')
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus // 2, mp_context=mp.get_context('forkserver'))
    # warc_filepaths = ["a.warc.gz", "b.warc.gz", "c.warc.gz"]
    s3 = s3fs.S3FileSystem()
    downloaded_files = [p.stem for p in (Path('data/warc-train').glob('*.filtered.txt'))]
    print(f'{len(downloaded_files)=}')
    warc_filepaths = [Path(p) for p in s3.ls('ai-residency-stanford-snap-uce/dataset/mds/CC') if Path(p).stem not in downloaded_files]
    print(warc_filepaths[:5])
    print(downloaded_files[:5])
    print(f'{len(warc_filepaths)=}')
    # warc_filepaths = list(Path('data/').glob('*.warc.filtered.gz'))
    output_directory_path = "./data/warc-train/"
    futures = []
    for i, warc_filepath in list(enumerate(warc_filepaths)):
        # For each WARC filepath, submit a job to the executor and get a future back
        warc_filename = warc_filepath.stem
        print(f'Queuing WARC file: {warc_filename}')
        out_filename = warc_filename + '.txt'
        future = executor.submit(
            process_single_warc_file,
            i,
            warc_filepath,
            os.path.join(output_directory_path, out_filename)
        )
        # Store the futures
        futures.append(future)
    # Iterate over the completed futures as they finish, using a progress bar # to keep track of progress.
    for future in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(warc_filepaths),
        smoothing=0.0,
        desc='Total WARC progress'
    ):
        output_file = future.result()
        print(f"Output file written: {output_file}")


if __name__ == '__main__':
    # train_fasttext_model(dataset_path='data/paloma-like-train.txt', model_path='data/paloma-like.bin', validation_path='data/paloma-like-train.txt')
    # run_filter('data/CC-MAIN-20180420081400-20180420101400-00118.warc.gz', limit=10_000)
    parallel_run_filter()