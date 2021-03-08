from utils import compute

compute.get_torch()
import json
import os
import sys

cwd = os.getcwd()
print('cwd is: ', cwd)
# sys.path.append(cwd[:cwd.index('pycharm_project_') + + len('pycharm_project_') + 4])

from haystack.retriever.dense import DensePassageRetriever
from haystack.document_store.faiss import FAISSDocumentStore

query_model = "facebook/dpr-question_encoder-single-nq-base"
passage_model = "facebook/dpr-ctx_encoder-single-nq-base"
save_dir = "../saved_models/dpr"

should_update_document_store = True
document_store_save_path = 'ds_save_file'

doc_dir = '/data/'

if should_update_document_store:

    # formated_file_name = doc_dir + 'startqa_corpus_formatted_for_documentstore.json'
    # TEMP FOR TESTING
    formated_file_name = doc_dir + 'sample_startqa_corpus_formatted_for_documentstore.jsonl'

    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")

    dicts = []
    max_docs = 10_00
    with open(formated_file_name, 'r') as corpus:
        counter = 1
        for line in corpus:
            if line.startswith('[') or line.startswith(']'):
                continue
            d = json.loads(line)
            if d['meta']['title']:
                d['meta']['name'] = d['meta']['title']
            dicts.append(d)
            counter += 1
            if counter % 1000 == 0:
                document_store.write_documents(dicts)
                dicts = []
                print('wrote ', counter, ' documents')
            if counter > max_docs:
                break
    print('done writing')
    dicts = []

    # dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
    # Now, let's write the dicts containing documents to our DB.
    doc_dir = "data/"
    print(document_store.get_all_documents())

    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model=query_model,
        passage_embedding_model=passage_model,
        max_seq_len_query=64,
        max_seq_len_passage=256,
        # use_gpu=False
    )

    print('updating document store')

    # this is done only once
    document_store.update_embeddings(retriever)
    document_store.save(document_store_save_path)
# loading existing store
else:
    document_store = FAISSDocumentStore.load(document_store_save_path)
