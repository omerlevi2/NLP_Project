from dpr.experiments.document_store import get_faiss_document_store, populate_document_store_from_strategyqa, \
    save_document_store
from dpr.retrievers.retrieves import get_retriever, save_retriever
from dpr.experiments.test_read_formatted_startqa_to_document_store import formated_file_name


def create_new_ds_and_new_retriever(name=formated_file_name):
    document_store = get_faiss_document_store()
    populate_document_store_from_strategyqa(name, document_store)
    retriever = get_retriever(document_store)
    print('updating document store')
    # this is done only once
    document_store.update_embeddings(retriever)
    save_document_store(document_store)
    save_retriever(retriever)
