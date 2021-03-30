from haystack.retriever.sparse import ElasticsearchRetriever

from dpr.retrievers.dataset.StrategyQADataset import StrategyQADataset


def get_all_positive_contexts(dataset):
    gold_paragraphs = set()
    for example in dataset:
        paragraphs = [x['text'] for x in example['positive_ctxs']]
        gold_paragraphs.update(paragraphs)
    return gold_paragraphs


train_para = get_all_positive_contexts(StrategyQADataset().train_set())
dev_para = get_all_positive_contexts(StrategyQADataset().dev_set())
all_paras = train_para.union(dev_para)

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
import concurrent.futures
import dpr.experiments.document_store as doc_store_utils
import random
from dpr.retrievers.corpus.StrategyQAWikiCorpus import StrategyQAWikiCorpus

elastic_ds = doc_store_utils.get_elastic_document_store()

num_existing_docs = elastic_ds.get_document_count()

if num_existing_docs != 0:
    raise Exception("elastic already populated")

batch_size = 1_000


def write_dicts(elastic, dicts):
    print('starting to write', len(dicts))
    elastic.write_documents(dicts, batch_size=batch_size)


total = 0

r = list(range(37))


def take_once_every_36():
    return random.choice(r) == 1


def should_take(x):
    return x['text'] in all_paras or take_once_every_36()


def get_dicts():
    for i, x in enumerate(StrategyQAWikiCorpus().iter_json_batches(batch_size=10_000, offset=0)):
        if not x:
            print('skipping ', x)
            continue
        print('iteration is ', i)
        yield [ex for ex in x if should_take(ex)]


print(total)
max_size = 9999999999999999
# dicts = []
futures = []

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(lambda dicts: write_dicts(elastic_ds, dicts),
                 get_dicts())

print('num docs', elastic_ds.get_document_count())
# print(elastic_ds.get_all_documents())
