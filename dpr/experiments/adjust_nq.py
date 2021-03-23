import concurrent
import json

from tqdm import tqdm

import dpr.experiments.document_store as document_store_utils
import time
from haystack.retriever.sparse import TfidfRetriever, ElasticsearchRetriever
import pylcs

from dpr.retrievers.dataset.NQDataset import NQDataset

document_store = document_store_utils.get_elastic_document_store()
print(document_store.get_document_count())

retriever = ElasticsearchRetriever(document_store=document_store)


def retrieve_inner(context, result):
    retrieve = retriever.retrieve(context, top_k=1)
    # print('searched for:', context)
    text = retrieve[0].text
    # print('found:', text)
    lcs = pylcs.lcs2(text, context)
    if lcs >= min(len(text), len(context)) * 0.5:
        result.append(context)


bar = tqdm(total=307_000)
took = 0
with open('new_nq_train.json', 'w') as new_nq_file:
    for i, sample in enumerate(NQDataset().train_set()):
        positive_contexts = [x['text'] for x in sample['positive_ctxs']]
        start_q = time.time()
        adjusted_contexts = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            executor.map(lambda context: retrieve_inner(context, adjusted_contexts), positive_contexts)
        if len(adjusted_contexts) > 0:
            took += 1
            bar.update(1)
            # TODO HERE ADD , NONE
            sample['negative_ctxs'] = []
            sample['positive_ctxs'] = adjusted_contexts
            new_nq_file.write(json.dumps(sample) + '\n')

        print('percent questions filtered', took / (i + 1))
        print('question', i, 'took', time.time() - start_q)
