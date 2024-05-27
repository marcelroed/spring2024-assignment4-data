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
    def __init__(self):
        self.language_model = IdentifyLanguageModel()
        self.nsfw_model = NSFWModel()
        self.toxic_model = ToxicSpeechModel()
        # self.quality_model = QualityModel()
        # self.paloma_like_model = PalomaLikeModel()
        self.gopher_model = GopherModel()
        self.stats = defaultdict(Stats)
    
    def __call__(self, text):
        for model_args in [(self.language_model, 'en', 0.9), (self.gopher_model, 'keep', 0.5), (self.nsfw_model, 'non-nsfw', 0.8), (self.toxic_model, 'non-toxic', 0.8)]:
            model_keep = self.use_model(*model_args, text)
            if not model_keep:
                return False
        return True
    
    def use_model(self, model, keep_label, keep_threshold, text):
        stats = self.stats[model.__class__.__name__]
        label, prob = model.predict(text)
        if label != keep_label:
            prob = 1 - prob
            label = keep_label
        
        if prob >= keep_threshold:
            stats.retained += 1
            return True
        else:
            stats.removed += 1
            return False
        
    def show_stats(self):
        for k, v in self.stats.items():
            print(f'{k}: {v.retained} retained, {v.removed} removed ({v.kept_ratio() * 100:.2f}% kept)')


def run_filter(warc_path: str | Path, limit=None):
    filterer = Filterer()
    out_texts = []
    for text in islice(iterwarc_text(warc_path), limit):
        if filterer(text):
            out_texts.append(text)
    filterer.show_stats()
    print(len(out_texts))


if __name__ == '__main__':
    # train_fasttext_model(dataset_path='data/paloma-like-dataset-train.txt', model_path='data/paloma-like.bin', validation_path='data/paloma-like-dataset-valid.txt')
    run_filter('data/CC-MAIN-20180420081400-20180420101400-00118.warc.gz', limit=10_000)