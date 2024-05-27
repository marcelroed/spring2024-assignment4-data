from pathlib import Path
from fasttext import train_supervised, load_model


def train_fasttext_model(dataset_path: str | Path, model_path: str | Path, validation_path: str | Path | None = None) -> None:
    """Train a quality classifier model on the given labeled text files."""
    model = train_supervised(input=str(dataset_path), epoch=50)
    model.save_model(str(model_path)) 
    # Also check that the model works
    if validation_path is not None:
        samples, precision, recall = model.test(validation_path)
        print(f'Precision: {precision}, Recall: {recall}')
    return model
    

class QualityModel:
    def __init__(self, model_path: str | Path):
        self.model = load_model(str(model_path))
    
    def predict(self, text: str):
        label, prob = self.model.predict(text.replace('\n', ' '))
        label = label[0].replace('__label__', '')
        if label == 'neg':
            label = 'cc'
        else:
            label = 'wiki'
        return label, prob[0]
        

def load_and_predict(text: str):
    model = QualityModel('data/fasttext-quality.bin')
    return model.predict(text)



if __name__ == '__main__':
    pass
    # train_fasttext_model(dataset_path='data/quality-dataset-train.txt', model_path='data/fasttext-quality.bin', validation_path='data/quality-dataset-valid.txt')