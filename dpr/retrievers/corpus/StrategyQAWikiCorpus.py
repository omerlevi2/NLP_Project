from haystack.preprocessor.utils import fetch_archive_from_http


class StrategyQAWikiCorpus:
    def __init__(self):
        s3_url_dev = 'link_to_my_zip_startqa_corpus_formatted_for_documentstore.zip'
        fetch_archive_from_http(s3_url_dev, output_dir='corpus/')
