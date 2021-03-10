import json

from haystack.document_store.faiss import FAISSDocumentStore
from tqdm import tqdm

from dpr.experiments import hyperparams

document_store_save_path = 'ds_save_file'

max_docs_to_write = 500
write_batch_size = 10_000

sql_url: str = "sqlite:///haystack_test_faiss.db"


def populate_document_store_from_strategyqa(formated_file_name, document_store):
    dicts = []

    with open(formated_file_name, 'r') as corpus:
        counter = 1
        for line in tqdm(corpus):
            if line.startswith('[') or line.startswith(']'):
                continue
            d = json.loads(line)
            if d['meta']['title']:
                d['meta']['name'] = d['meta']['title']
            dicts.append(d)
            counter += 1
            if counter % write_batch_size == 0:
                document_store.write_documents(dicts)
                dicts = []
                print('wrote ', counter, ' documents')
            if counter > max_docs_to_write:
                break
    document_store.write_documents(dicts)
    print('done writing')
    assert document_store.get_document_count() > 0


def load_saved_document_store(document_store_class=FAISSDocumentStore):
    return document_store_class.load(document_store_save_path, sql_url=sql_url)


def get_faiss_document_store():
    return FAISSDocumentStore(faiss_index_factory_str=hyperparams.faiss_index_factory_str, sql_url=sql_url)


def save_document_store(document_store, path=document_store_save_path):
    document_store.save(path)
