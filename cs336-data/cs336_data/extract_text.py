from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import gzip
from fastwarc import ArchiveIterator, WarcRecord

def extract_text_from_html_bytes(s: bytes) -> str:
    encoding = detect_encoding(s)
    decoded = s.decode(encoding, errors='replace')
    return extract_plain_text(decoded)


def iterwarc(filename: str):
    """Iterate responses in a compressed WARC file."""
    with open(filename, 'rb') as f:
        for record in ArchiveIterator(f):
            if record.headers['WARC-Type'] == 'response':
                yield record


def run_extract_file(filename: str, limit=3) -> str:
    outputs = []
    for record in iterwarc(filename):
        text = extract_text_from_html_bytes(record.reader.read())
        outputs.append(text)
        if limit is not None and len(outputs) >= limit:
            break
    return outputs
    #     html_bytes = f.read()
    #     text = extract_text_from_html_bytes(html_bytes)
    # print(text)

def main():
    result = run_extract_file('data/CC-MAIN-20180420081400-20180420101400-00118.warc.gz')
    print(result)

if __name__ == '__main__':
    main()