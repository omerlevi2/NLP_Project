import json

from dpr.experiments import document_store
from dpr.retrievers import retrieves
from dpr.retrievers.dataset.StrategyQADataset import StrategyQADataset

qa_dataset = StrategyQADataset()
ds = document_store.get_elastic_document_store()
retriever = retrieves.load_retriever(ds, 'ret_save_file_nq_then_strat')


def retrieve_passages(dataset, d):
    for example in dataset:
        question = example['question']
        retrieved = retriever.retrieve(question, top_k=7)
        # passages = [x.text for x in retrieved]
        print(example)
        d[example] = retrieved


result = {}
retrieve_passages(qa_dataset.train_set(), result)
retrieve_passages(qa_dataset.dev_set(), result)
print(result)

with open('strategy_retrieved_passages.json', 'w', encoding='utf-8') as f:
    json.dump(result, f)

