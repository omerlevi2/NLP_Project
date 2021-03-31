from haystack.preprocessor.utils import fetch_archive_from_http
import json


class NQDataset:
    def __init__(self):
        self.s3_url_train = 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz'
        self.s3_url_dev = 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz'
        fetch_archive_from_http(self.s3_url_dev, output_dir='corpus/dev')
        fetch_archive_from_http(self.s3_url_train, output_dir='corpus/train')

        self.data_dir = 'corpus'
        self.train_filename = 'train/biencoder-nq-train.json'
        self.dev_filename = 'dev/biencoder-nq-dev.json'

    def train_set(self):
        return self._iter_set('corpus/%s' % self.train_filename)

    def dev_set(self):
        return self._iter_set('corpus/%s' % self.dev_filename)

    def _iter_set(self, dpr_train_split):
        stack = 0
        read = ''
        with open(dpr_train_split, 'r') as corpus:
            for line in corpus:
                if line.startswith('[') or line.startswith(']'):
                    continue
                line = line.lstrip()
                if line.startswith('}'):
                    stack -= 1
                    if stack == 0:
                        line = line.replace(',', '')
                if line.startswith('{'):
                    stack += 1
                read += line
                if stack == 0:
                    try:
                        d = json.loads(read)
                        read = ''
                        yield d
                    except Exception:
                        # print('fail parsing to json ', read)
                        print('fail parsing to json ')
                        read = ''
                    #     continue
                # if d['meta']['title']:
                #     d['meta']['name'] = d['meta']['title']
