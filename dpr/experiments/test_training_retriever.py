import dpr.retrievers.retrieves as retrieves
import dpr.retrievers.trainer as trainer
from dpr.experiments import flows
from dpr.retrievers.dataset.StrategyQADataset import StrategyQADataset
from dpr.retrievers.trainer import RetrieverTrainParams
import document_store
from dpr.experiments.flows import update_document_store_embeddings_and_save

load = False
populate_doucment_store = True
train = False

doc_dir = 'data/'
formated_file_name = doc_dir + 'startqa_corpus_formatted_for_documentstore.json'
# formated_file_name = doc_dir + 'sample_startqa_corpus_formatted_for_documentstore.jsonl'


if load:
    ds = document_store.load_saved_document_store()
    retriever = retrieves.load_retriever(ds)
else:
    # here possibliy create from scratch.
    ds = document_store.get_faiss_document_store()
    print('document count:', ds.get_document_count())
    # if populate_doucment_store:
    #     ds = flows.create_faiss_db_on_stratqa_corpus()
    if train:
        retriever = retrieves.get_retriever_for_training()
        trainer.train(retriever,
                      StrategyQADataset(),
                      RetrieverTrainParams(num_hard_negatives=0, n_epochs=1, batch_size=1), save=False)
        update_document_store_embeddings_and_save(ds, retriever)
        ds = document_store.load_saved_document_store()
    else:
        retriever = retrieves.load_retriever(ds)

res = retriever.retrieve('West')
print(res)
print(ds.get_document_count())
