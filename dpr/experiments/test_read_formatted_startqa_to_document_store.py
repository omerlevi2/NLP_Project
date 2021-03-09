# from utils import compute
#
# compute.get_torch()

from haystack.retriever.dense import DensePassageRetriever
from haystack.document_store.faiss import FAISSDocumentStore

from dpr.experiments.startQA import populate_document_store_from_startqa

query_model = "facebook/dpr-question_encoder-single-nq-base"
passage_model = "facebook/dpr-ctx_encoder-single-nq-base"

should_update_document_store = True
document_store_save_path = 'ds_save_file'

doc_dir = 'data/'

if should_update_document_store:

    formated_file_name = doc_dir + 'startqa_corpus_formatted_for_documentstore.json'
    # TEMP FOR TESTING
    # formated_file_name = doc_dir + 'sample_startqa_corpus_formatted_for_documentstore.jsonl'

    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")

    populate_document_store_from_startqa(formated_file_name, document_store)

    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model=query_model,
        passage_embedding_model=passage_model,
        max_seq_len_query=64,
        max_seq_len_passage=256,
        use_gpu=True
    )

    print('updating document store')

    # this is done only once
    document_store.update_embeddings(retriever)
    document_store.save(document_store_save_path)
    print('done')
# loading existing store
else:
    document_store = FAISSDocumentStore.load(document_store_save_path)
