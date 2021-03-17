import dpr.retrievers.retrieves as retrieves
import dpr.retrievers.trainer as trainer
import document_store
from dpr.experiments.flows import update_document_store_embeddings_and_save

load = False

doc_dir = 'data/'
formated_file_name = doc_dir + 'startqa_corpus_formatted_for_documentstore.json'
# formated_file_name = doc_dir + 'sample_startqa_corpus_formatted_for_documentstore.jsonl'

if load:
    ds = document_store.load_saved_document_store()
    retriever = retrieves.load_retriever(ds)
else:
    # here possibliy create from scratch.
    ds = document_store.get_faiss_document_store()
    retriever = retrieves.get_retriever_for_training()
    trainer.train(retriever,
                  trainer.RetrieverTrainParams(num_hard_negatives=0, n_epochs=1, batch_size=1), save=False)
    update_document_store_embeddings_and_save(ds, retriever)
    ds = document_store.load_saved_document_store()

res = retriever.retrieve('West')
print(res)
print(ds.get_document_count())
