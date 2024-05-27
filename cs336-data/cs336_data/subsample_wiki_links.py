from random import sample

def ceildiv(a, b):
    return -(-a // b)


def subsample_wiki_links():
    with open('data/enwiki-20240420-extracted_urls.txt', 'r') as f:
        urls = f.read().splitlines()
        urls = [url for url in urls if not url.endswith('.pdf') and not url.endswith('.eps')]
        print(urls[:10])
        samples = sample(urls, 100_000)
        print(samples[:10])
        with open('data/enwiki-samples-100k.txt', 'w') as f:
            f.write('\n'.join(samples))

def split_subsample():
    """Split the subsampled wiki links into """
    n_chunks = 32
    with open('data/enwiki-samples-100k.txt', 'r') as f:
        lines = f.read().splitlines()
        chunk_size = ceildiv(len(lines), 32)
        for i in range(n_chunks):
            with open(f'data/enwiki-samples-100k-chunk-{i:02}.txt', 'w') as out_f:
                out_f.write('\n'.join(lines[i * chunk_size: (i + 1) * chunk_size]))

if __name__ == '__main__':
    # subsample_wiki_links()
    split_subsample()

