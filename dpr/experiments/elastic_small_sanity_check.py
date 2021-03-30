from haystack.retriever.sparse import ElasticsearchRetriever

from dpr.retrievers.dataset.StrategyQADataset import StrategyQADataset


def get_all_positive_contexts(dataset):
    gold_paragraphs = set()
    for example in dataset:
        paragraphs = [x['text'] for x in example['positive_ctxs'] if len(x['text']) < 1200]
        gold_paragraphs.update(paragraphs)
    return gold_paragraphs


train_para = get_all_positive_contexts(StrategyQADataset().train_set())
dev_para = get_all_positive_contexts(StrategyQADataset().dev_set())
all_paras = train_para.union(dev_para)

import dpr.experiments.document_store as doc_store_utils

elastic_ds = doc_store_utils.get_elastic_document_store()
retriever = ElasticsearchRetriever(document_store=elastic_ds)
mistakes = 0
for i, s in enumerate(all_paras):
    retrieve = retriever.retrieve(s, top_k=1)[0].text
    if not s == retrieve:
        print('expected ', s)
        print('got', retrieve)
        retrieve = [x.text for x in retriever.retrieve(s, top_k=20)]
        retrieve = [x for x in retrieve if x == s]
        if retrieve:
            print('found on second try', retrieve)
            continue
        mistakes += 1
        print('mistakes', mistakes)
        print('total', i + 1)
