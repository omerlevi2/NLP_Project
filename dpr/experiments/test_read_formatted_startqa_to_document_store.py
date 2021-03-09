# from utils import compute
#
# compute.get_torch()

from haystack.document_store.faiss import FAISSDocumentStore

from dpr.experiments.retriever import get_retriever
from dpr.experiments.document_store import populate_document_store_from_strategyqa


should_update_document_store = True
document_store_save_path = 'ds_save_file'

doc_dir = 'data/'

if should_update_document_store:

    formated_file_name = doc_dir + 'startqa_corpus_formatted_for_documentstore.json'
    # TEMP FOR TESTING
    # formated_file_name = doc_dir + 'sample_startqa_corpus_formatted_for_documentstore.jsonl'

    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")

    populate_document_store_from_strategyqa(formated_file_name, document_store)

    retriever = get_retriever(document_store)

    print('updating document store')

    # this is done only once
    document_store.update_embeddings(retriever)
    document_store.save(document_store_save_path)
    print('done')
# loading existing store
else:
    document_store = FAISSDocumentStore.load(document_store_save_path)
