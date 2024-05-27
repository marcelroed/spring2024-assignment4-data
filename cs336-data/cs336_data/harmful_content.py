from fasttext import load_model
from cs336_data.extract_text import iterwarc, extract_text_from_html_bytes

class NSFWModel:
    def __init__(self, path='data/jigsaw_fasttext_bigrams_nsfw_final.bin'):
        self.model = load_model(path)

    def predict(self, text):
        label, prob = self.model.predict(text.replace('\n', ' '))
        label = label[0].replace('__label__', '')
        return label, prob[0]


class ToxicSpeechModel:
    def __init__(self, path='data/jigsaw_fasttext_bigrams_hatespeech_final.bin'):
        self.model = load_model(path)
    
    def predict(self, text):
        label, prob = self.model.predict(text.replace('\n', ' '))
        label = label[0].replace('__label__', '')
        return label, prob[0]


def run_classify_samples():
    nsfw_model = NSFWModel()
    toxic_model = ToxicSpeechModel()
    for record in iterwarc('data/CC-MAIN-20180420081400-20180420101400-00118.warc.gz'):
        text = extract_text_from_html_bytes(record.reader.read())
        nsfw_label, nsfw_prob = nsfw_model.predict(text)
        toxic_label, toxic_prob = toxic_model.predict(text)
        if (nsfw_label == 'nsfw' and nsfw_prob > 0.5): # or (toxic_label == 'toxic' and toxic_prob > 0.5):
            print(text[:2000])
            print(nsfw_label, nsfw_prob, toxic_label, toxic_prob)
            input()


def main():
    run_classify_samples()
    



if __name__ == '__main__':
    main()