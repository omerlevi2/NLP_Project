from dpr.experiments.document_store import populate_document_store_from_strategyqa
from haystack.document_store.sql import SQLDocumentStore
from haystack.retriever.sparse import TfidfRetriever

from dpr.retrievers.corpus.StrategyQAWikiCorpus import StrategyQAWikiCorpus
from dpr.retrievers.dataset.NQDataset import NQDataset

formated_file_name = StrategyQAWikiCorpus().filepath()

document_store = SQLDocumentStore(url="sqlite:///qa.db")
print(document_store.get_document_count())
if document_store.get_document_count() == 0:
    populate_document_store_from_strategyqa(formated_file_name, document_store)

retriever = TfidfRetriever(document_store=document_store)
for sample in NQDataset().train_set():
    question_ = sample['question']
    print(question_)
    # print(sample['positive_ctxs'][0]['text'])
    retrieve = retriever.retrieve(question_, top_k=1)
    print(retrieve)
