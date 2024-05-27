from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

from fasttext import load_model

from cs336_data.deduplication import minhash_deduplication
from cs336_data.extract_text import extract_text_from_html_bytes, iterwarc
from cs336_data.harmful_content import NSFWModel, ToxicSpeechModel
from cs336_data.identify_language import IdentifyLanguageModel
from cs336_data.quality_classifier import QualityModel, train_fasttext_model
from cs336_data.quality_filter import gopher_quality_filter
    

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
        self.nsfw_model = NSFWModel()
        self.toxic_model = ToxicSpeechModel()
        self.quality_model = QualityModel('data/fasttext-quality.bin')
        self.language_model = IdentifyLanguageModel()
        self.paloma_like_model = PalomaLikeModel()
        self.stats = defaultdict(Stats)
    
    def __call__(self, record):
        text = extract_text_from_html_bytes(record.reader.read())
        language
    
    def use_model(self, model, keep_label, keep_threshold, text):
        stats = self.stats[self.model.__class__.__name__]
        label, prob = model.predict(text)
        if label != keep_label:
            prob = 1 - prob
            label = keep_label
        
        if prob > keep_threshold:
            stats.retained += 1
            return True
        else:
            stats.removed += 1
            return False
        
        



if __name__ == '__main__':
    train_fasttext_model(dataset_path='data/paloma-like-dataset-train.txt', model_path='data/paloma-like.bin', validation_path='data/paloma-like-dataset-valid.txt')