from cs336_data.extract_text import extract_text_from_html_bytes, iterwarc
import nltk
from logging import getLogger, basicConfig

logger = getLogger(__name__)

def gopher_quality_filter(s: str) -> bool:
    """Return True if the input string passes the quality filter, False otherwise."""
    nltk.download('punkt', quiet=True)
    tokens = nltk.word_tokenize(s)

    num_tokens = len(tokens)
    if num_tokens < 50 or num_tokens > 100_000:
        # logger.info(f'Number of tokens {num_tokens} is out of range')
        return False

    mean_word_length = sum(len(token) for token in tokens) / num_tokens
    if mean_word_length < 3 or mean_word_length > 10:
        # logger.info(f'Mean word length {mean_word_length} is out of range')
        return False
    
    end_ellipsis = [line.endswith('...') for line in s.splitlines()]
    proportion_end_ellipsis = sum(end_ellipsis) / len(end_ellipsis)
    if proportion_end_ellipsis > 0.3:
        # logger.info(f'Proportion of lines ending in ellipsis {proportion_end_ellipsis} is too high')
        return False
    
    contains_alphanum = [any(c.isalnum() for c in t) for t in tokens]
    contains_alphanum_proportion = sum(contains_alphanum) / num_tokens
    if contains_alphanum_proportion < 0.8:
        # logger.info(f'Proportion of tokens containing alphanumeric characters {contains_alphanum_proportion} is too low')
        return False
    
    return True

def run_filter_samples():
    for record in iterwarc('data/CC-MAIN-20180420081400-20180420101400-00118.warc.gz'):
        text = extract_text_from_html_bytes(record.reader.read())
        print(text)
        should_keep = gopher_quality_filter(text)
        print(f'{should_keep=}')
        input()

def main():
    run_filter_samples()


if __name__ == '__main__':
    basicConfig(level='INFO')
    main()