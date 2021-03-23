import json
from haystack.preprocessor.utils import fetch_archive_from_http


class StrategyQAWikiCorpus:
    def __init__(self):
        s3_url_dev = 'https://dpr-nlp.s3.amazonaws.com/startqa_corpus_formatted_for_documentstore.zip'
        fetch_archive_from_http(s3_url_dev, output_dir='corpus/stratCorpus')

    def filepath(self):
        return 'corpus/stratCorpus/startqa_corpus_formatted_for_documentstore.json'

    def iter_jsons(self, offset=0):
        with open(self.filepath(), 'r') as corpus:
            for i, line in enumerate(corpus):
                if line.startswith('[') or line.startswith(']'):
                    continue
                if i < offset:
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

    def iter_json_batches(self, batch_size=10_000, offset=0, max_size=999999999999):
        dicts = []
        for i, json in enumerate(self.iter_jsons(offset=offset), start=1):
            dicts.append(json)
            if i % batch_size == 0:
                yield dicts
                dicts = []
            if i > max_size:
                break
        yield dicts
