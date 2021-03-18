from dpr.experiments import document_store
from dpr.experiments.document_store import get_faiss_document_store, populate_document_store_from_strategyqa, \
    save_document_store
# from dpr.experiments.loading_nq_flow import document_store
from dpr.retrievers.corpus.StrategyQAWikiCorpus import StrategyQAWikiCorpus
from dpr.retrievers.retrieves import get_retriever, save_retriever


# from dpr.experiments.test_read_formatted_startqa_to_document_store import formated_file_name


def create_new_ds_and_new_retriever(corpus_filename):
    document_store = get_faiss_document_store()
    populate_document_store_from_strategyqa(corpus_filename, document_store)
    retriever = get_retriever(document_store)
    print('updating document store')
    # this is done only once
    update_document_store_embeddings_and_save(document_store, retriever)


def update_document_store_embeddings_and_save(document_store, retriever):
    document_store.update_embeddings(retriever)
    save_document_store(document_store)
    save_retriever(retriever)


def create_faiss_db_on_stratqa_corpus():
    ds = document_store.get_faiss_document_store()
    print('document count:', ds.get_document_count())
    if ds.get_document_count() == 0:
        path = StrategyQAWikiCorpus().filepath()
        document_store.populate_document_store_from_strategyqa(path, ds)

    return ds
