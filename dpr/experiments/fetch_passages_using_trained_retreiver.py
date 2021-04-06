import json

from tqdm import tqdm

from dpr.experiments import document_store
from dpr.retrievers import retrieves
from dpr.retrievers.dataset.StrategyQADataset import StrategyQADataset
import torch

qa_dataset = StrategyQADataset()
ds = document_store.get_elastic_document_store()
retriever = retrieves.load_retriever(ds, 'ret_save_file_nq_then_strat')


# embeded = retriever.embed_queries(['Who shot yeah'])
#
# docs = ds.get_all_documents()
# passage_embeddings = torch.tensor([doc.embedding for doc in docs])
# print(ds.query('Who are you'))
#
# passage_embeddings = torch.tensor([doc.embedding for doc in docs])  # shape(n_passage,d)
# questions = [x['question'] for x in qa_dataset.train_set()] + [x['question'] for x in qa_dataset.dev_set()]
# question_embeddings = retriever.embed_queries(questions).T  # shape (d,n_questions)
# result = torch.matmul(passage_embeddings, question_embeddings)  # shape (n_passage,n_questions
# # top_k indexes at column x are indexes in docs for questions[x]
# values, indices, = torch.topk(result, k=7, dim=0)
# print(indices)


# _,top_k = torch.topk(result,k=7)
# print(result)

def retrieve_passages(dataset, d):
    for example in tqdm(dataset):
        question = example['question']
        # embeded = retriever.embed_queries([question])
        # embeded = embeded[0]
        retrieved = retriever.retrieve(question, top_k=7)
        ans = []
        for ret in retrieved:
            ret_dict = ret.__dict__
            ret_dict.pop('embedding', None)
            ans.append(ret_dict)
        # passages = [x.text for x in retrieved]
        # scores = passage_embeddings * embeded
        print(question)
        print(ans)
        print('\n\n')
        d[question] = ans
        # break


result = {}
retrieve_passages(qa_dataset.train_set(), result)
retrieve_passages(qa_dataset.dev_set(), result)
print(result)

with open('strategy_retrieved_passages.json', 'w', encoding='utf-8') as f:
    json.dump(result, f)
