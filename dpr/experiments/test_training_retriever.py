import dpr.retrievers.retrieves as retrieves
import dpr.retrievers.trainer as trainer
import document_store

load = True

doc_dir = 'data/'
formated_file_name = doc_dir + 'startqa_corpus_formatted_for_documentstore.json'
# formated_file_name = doc_dir + 'sample_startqa_corpus_formatted_for_documentstore.jsonl'

if load:
    ds = document_store.get_faiss_document_store()
    retriever = retrieves.load_retriever(ds)
else:
    retriever = retrieves.get_retriever_for_training()
    trainer.train(retriever,
                  trainer.RetrieverTrainParams(num_hard_negatives=0, n_epochs=1))

res = retriever.retrieve('West')
print(res)
print(ds.get_document_count())
print(3)
