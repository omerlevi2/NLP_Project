import dpr.retrievers.retrieves as retrieves
import dpr.retrievers.trainer as trainer
from dpr.retrievers.dataset.NQDataset import NQDataset
from dpr.retrievers.dataset.StrategyQADataset import StrategyQADataset
from dpr.retrievers.trainer import RetrieverTrainParams
import document_store
from dpr.experiments.flows import update_document_store_embeddings_and_save

load = False
populate_doucment_store = False
train = True
update_embeddings = False

doc_dir = 'data/'
formated_file_name = doc_dir + 'startqa_corpus_formatted_for_documentstore.json'
# formated_file_name = doc_dir + 'sample_startqa_corpus_formatted_for_documentstore.jsonl'


if load:
    ds = document_store.load_saved_document_store()
    retriever = retrieves.load_retriever(ds)
else:
    # ds = document_store.get_elastic_document_store()
    # print('document count:', ds.get_document_count())
    if train:
        qa_dataset = StrategyQADataset()
        # qa_dataset = NQDataset()
        retriever = retrieves.get_retriever_for_training()
        trainer.train(retriever,
                      qa_dataset,
                      RetrieverTrainParams(num_positives=1, num_hard_negatives=1, n_epochs=1,
                                           batch_size=16,
                                           grad_acc_steps=8, evaluate_every=100), save=True)
    else:
        retriever = retrieves.load_retriever(ds)
    if update_embeddings:
        update_document_store_embeddings_and_save(ds, retriever)
    # ds = document_store.load_saved_document_store()

res = retriever.retrieve('"who sings does he love me with reba')
print(res)
print(ds.get_document_count())
