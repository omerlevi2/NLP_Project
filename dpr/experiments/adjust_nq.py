import concurrent

from dpr.experiments.document_store import populate_document_store_from_strategyqa
import dpr.experiments.document_store as document_store_utils
import time
from haystack.retriever.sparse import TfidfRetriever, ElasticsearchRetriever
import pylcs

from dpr.retrievers.corpus.StrategyQAWikiCorpus import StrategyQAWikiCorpus
from dpr.retrievers.dataset.NQDataset import NQDataset

formated_file_name = StrategyQAWikiCorpus().filepath()

# document_store = SQLDocumentStore(url="sqlite:///qa.db")
document_store = document_store_utils.get_elastic_document_store()
print(document_store.get_document_count())
if document_store.get_document_count() == 0:
    populate_document_store_from_strategyqa(formated_file_name, document_store)

print('initing tfidretriver')
retriever = ElasticsearchRetriever(document_store=document_store)
took = 0
total = 0


def retrieve_inner(context):
    global start, retrieve
    # total += 1
    start = time.time()
    retrieve = retriever.retrieve(context, top_k=1)
    print('searched for:', context)
    text = retrieve[0].text
    print('found:', text)
    lcs = pylcs.lcs2(text, context)
    if lcs >= min(len(text), len(context)) * 0.5:
        # took += 1
        print('take')
    else:
        print('drop')
    # return


for sample in NQDataset().train_set():
    positive_contexts = [x['text'] for x in sample['positive_ctxs']]
    start_q = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        executor.map(retrieve_inner, positive_contexts)
    print('question took', time.time() - start_q)
    # prob = retrieve[0].probability
    # print('prob', prob)
    # print('score', retrieve[0].score)
    # print('took', time.time() - start)

    print('\n\n')


print('taking', took / total)

print('-' * 50, '\n\n')
