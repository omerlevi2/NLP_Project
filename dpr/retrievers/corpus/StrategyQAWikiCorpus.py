from haystack.preprocessor.utils import fetch_archive_from_http


class StrategyQAWikiCorpus:
    def __init__(self):
        s3_url_dev = 'https://dpr-nlp.s3.amazonaws.com/startqa_corpus_formatted_for_documentstore.zip'
        fetch_archive_from_http(s3_url_dev, output_dir='corpus/stratCorpus')

    def filepath(self):
        return 'corpus/stratCorpus/startqa_corpus_formatted_for_documentstore.json'
