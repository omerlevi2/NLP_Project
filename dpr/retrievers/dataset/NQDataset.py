from haystack.preprocessor.utils import fetch_archive_from_http


class NQDataset:
    def __init__(self):
        s3_url_train = 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz'
        s3_url_dev = 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz'
        fetch_archive_from_http(s3_url_dev, output_dir='corpus/dev')
        fetch_archive_from_http(s3_url_train, output_dir='corpus/train')


