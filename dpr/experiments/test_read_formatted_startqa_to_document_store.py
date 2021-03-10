from dpr.experiments.retriever import *
from dpr.experiments.document_store import populate_document_store_from_strategyqa, document_store_save_path, \
    load_saved_document_store, get_faiss_document_store

should_update_document_store = False

doc_dir = 'data/'
formated_file_name = doc_dir + 'startqa_corpus_formatted_for_documentstore.json'
# formated_file_name = doc_dir + 'sample_startqa_corpus_formatted_for_documentstore.jsonl'

if should_update_document_store:
    document_store = get_faiss_document_store()
    populate_document_store_from_strategyqa(formated_file_name, document_store)
    retriever = get_retriever(document_store)

    print('updating document store')
    # this is done only once
    document_store.update_embeddings(retriever)
    document_store.save(document_store_save_path)
    save_retriever(retriever)

    print('done')

# loading existing store
else:
    document_store = load_saved_document_store()
    retriever = load_retriever(document_store)
    print('1')
