from haystack.preprocessor.utils import fetch_archive_from_http


class WikiCorpus:
    def __init__(self):
        s3_url_train = "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz"
        fetch_archive_from_http(s3_url_train, output_dir='corpus/')
