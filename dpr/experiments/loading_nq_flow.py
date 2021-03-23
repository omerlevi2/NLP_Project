from dpr.experiments.document_store import populate_document_store_from_strategyqa
import dpr.experiments.document_store as document_store_utils
import time
from haystack.retriever.sparse import TfidfRetriever, ElasticsearchRetriever

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
for sample in NQDataset().train_set():
    question_ = sample['question']
    print(question_)
    # print(sample['positive_ctxs'][0]['text'])
    start = time.time()
    retrieve = retriever.retrieve(question_, top_k=1)
    print('took', time.time() - start)
    print(retrieve)
