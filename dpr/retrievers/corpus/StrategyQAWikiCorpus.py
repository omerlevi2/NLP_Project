import json
from haystack.preprocessor.utils import fetch_archive_from_http


class StrategyQAWikiCorpus:
    def __init__(self):
        s3_url_dev = 'https://dpr-nlp.s3.amazonaws.com/startqa_corpus_formatted_for_documentstore.zip'
        fetch_archive_from_http(s3_url_dev, output_dir='corpus/stratCorpus')

    def filepath(self):
        return 'corpus/stratCorpus/startqa_corpus_formatted_for_documentstore.json'

    def iter_jsons(self):
        with open(self.filepath(), 'r') as corpus:
            for line in corpus:
                if line.startswith('[') or line.startswith(']'):
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    print('fail parsing to json ', line)
                    continue
                if len(d['text']) > 1200:
                    continue
                if d['meta']['title']:
                    d['meta']['name'] = d['meta']['title']
                yield d

            yield d
