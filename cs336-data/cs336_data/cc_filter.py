from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from itertools import islice

from fasttext import load_model

from cs336_data.deduplication import minhash_deduplication
from cs336_data.extract_text import extract_text_from_html_bytes, iterwarc_text
from cs336_data.harmful_content import NSFWModel, ToxicSpeechModel
from cs336_data.identify_language import IdentifyLanguageModel
from cs336_data.quality_classifier import QualityModel, train_fasttext_model
from cs336_data.quality_filter import gopher_quality_filter, GopherModel

    

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
        # self.paloma_like_model = PalomaLikeModel()
        # self.quality_model = QualityModel()
        self.gopher_model = GopherModel()
        self.stats = defaultdict(Stats)
        self.model_args_list = [
            (self.language_model, 'en', 0.9), (self.gopher_model, 'keep', 0.5), (self.nsfw_model, 'non-nsfw', 0.8), (self.toxic_model, 'non-toxic', 0.8)
        ]
        if include_paloma:
            self.model_args_list.append((self.paloma_like_model, 'paloma', 0.5))
    
    def __call__(self, text):
        for model_args in self.model_args_list:
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


def run_filter(warc_path: str | Path, limit=None):
    filterer = Filterer(include_paloma=True)
    out_texts = []
    for text in islice(iterwarc_text(warc_path), limit):
        if filterer(text):
            out_texts.append(text)
    print(f'WARC: {warc_path.name}\n{filterer.show_stats()}')
    return out_texts


def parallel_run_filter():
    import concurrent.futures
    import os
    from tqdm import tqdm
    def process_single_warc_file(input_path: str, output_path: str):
        # TODO: read input path, process the input, and write the output to output_path
        return output_path
    # Set up the executor
    num_cpus = len(os.sched_getaffinity(0))
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
    warc_filepaths = ["a.warc.gz", "b.warc.gz", "c.warc.gz"]
    output_directory_path = "/path/to/output_directory/"
    futures = []
    for warc_filepath in warc_filepaths:
        # For each WARC filepath, submit a job to the executor and get a future back
        warc_filename = str(Path(warc_filepath).name)
        future = executor.submit(
            process_single_warc_file,
            warc_filepath,
            os.path.join(output_directory_path, warc_filepath)
        )
    # Store the futures
    futures.append(future)
    # Iterate over the completed futures as they finish, using a progress bar # to keep track of progress.
    for future in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(warc_filepaths),
    ):
        output_file = future.result()
        print(f"Output file written: {output_file}")


if __name__ == '__main__':
    train_fasttext_model(dataset_path='data/paloma-like-train.txt', model_path='data/paloma-like.bin', validation_path=None)
    # run_filter('data/CC-MAIN-20180420081400-20180420101400-00118.warc.gz', limit=10_000)