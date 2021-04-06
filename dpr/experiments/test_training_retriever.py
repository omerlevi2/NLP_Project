import json

import dpr.retrievers.retrieves as retrieves
import dpr.retrievers.trainer as trainer
from dpr.retrievers.dataset.NQDataset import NQDataset
from dpr.retrievers.dataset.StrategyQADataset import StrategyQADataset
from dpr.retrievers.trainer import RetrieverTrainParams
import document_store
from dpr.experiments.flows import update_document_store_embeddings_and_save

load = False
populate_doucment_store = False
train = False
update_embeddings = True
answer_flow = True

doc_dir = 'data/'
formated_file_name = doc_dir + 'startqa_corpus_formatted_for_documentstore.json'
# formated_file_name = doc_dir + 'sample_startqa_corpus_formatted_for_documentstore.jsonl'


if load:
    ds = document_store.load_saved_document_store()
    retriever = retrieves.load_retriever(ds)
else:
    ds = document_store.get_elastic_document_store()
    print('document count:', ds.get_document_count())
    if train:

        qa_dataset = StrategyQADataset()
        # qa_dataset = NQDataset()
        # retriever = retrieves.get_retriever_for_training()
        retriever = retrieves.load_retriever(ds, 'ret_save_file_nq_only')
        batch_size = 6
        trainer.train(retriever,
                      qa_dataset,
                      RetrieverTrainParams(num_positives=1, num_hard_negatives=5, n_epochs=2,
                                           batch_size=batch_size,
                                           grad_acc_steps=16 // batch_size, evaluate_every=500 // batch_size)
                      # RetrieverTrainParams(num_positives=1*4, num_hard_negatives=15, n_epochs=6,
                      #                      batch_size=16*4,
                      #                      grad_acc_steps=4, evaluate_every=100)
                      , save=True)
    else:
        retriever = retrieves.load_retriever(ds,'ret_save_file_nq_then_strat')
    if update_embeddings:
        update_document_store_embeddings_and_save(ds, retriever)
    # ds = document_store.load_saved_document_store()
    if answer_flow:
        qa_dataset = StrategyQADataset()

        def retrieve_passages(dataset, d):
            for example in dataset:
                question = example['question']
                # embeded = retriever.embed_queries([question])
                # embeded = embeded[0]
                retrieved = retriever.retrieve(question, top_k=7)
                # passages = [x.text for x in retrieved]
                print(example)
                d[question] = retrieved


        result = {}
        retrieve_passages(qa_dataset.train_set(), result)
        retrieve_passages(qa_dataset.dev_set(), result)
        print(result)

        with open('strategy_retrieved_passages.json', 'w', encoding='utf-8') as f:
            json.dump(result, f)

res = retriever.retrieve('"who sings does he love me with reba')
print(res)
print(ds.get_document_count())
