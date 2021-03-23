import json
import os
import time
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.document_store.faiss import FAISSDocumentStore

from dpr.experiments import hyperparams

document_store_save_path = 'ds_save_file'

max_docs_to_write = 5000_000_000_000
write_batch_size = 500_000
print_every = 100_000

sentence_len_threshold = 1200

# sql_url: str = "sqlite:///haystack_test_faiss.db"
sql_url: str = "sqlite:///haystack_stratqa-faiss.db"


def populate_document_store_from_strategyqa(formated_file_name, document_store):
    print('starting')
    with open(formated_file_name, 'r') as corpus:

        def iter_jsons():
            dicts = []
            counter = 1
            for line in corpus:
                if line.startswith('[') or line.startswith(']'):
                    continue
                counter += 1
                try:
                    d = json.loads(line)
                except Exception:
                    print('fail parsing to json ', line)
                    continue
                if len(d['text']) > sentence_len_threshold:
                    continue
                if d['meta']['title']:
                    d['meta']['name'] = d['meta']['title']
                dicts.append(d)

                if counter % print_every == 0:
                    print('wrote ', counter, ' documents')
                if counter % write_batch_size == 0:
                    write_to_document_store(dicts)
                    # yield dicts
                    dicts = []
                if counter > max_docs_to_write:
                    # yield dicts
                    write_to_document_store(dicts)
                    # dicts = []
                    break
            # yield dicts
            write_to_document_store(dicts)

        def write_to_document_store(dicts):
            print('writings ', len(dicts), ' documents')
            document_store.write_documents(dicts)

        iter_jsons()

        # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        #     executor.map(write_to_document_store, iter_jsons())
    # for x in iter_jsons():
    #     write_to_document_store(x)

    print('done writing')
    assert document_store.get_document_count() > 0


def load_saved_document_store(faiss_index_path=document_store_save_path, document_store_class=FAISSDocumentStore,
                              sql_url=sql_url):
    return document_store_class.load(faiss_index_path, sql_url=sql_url)


def get_faiss_document_store():
    return FAISSDocumentStore(faiss_index_factory_str=hyperparams.faiss_index_factory_str, sql_url=sql_url)


def save_document_store(document_store, path=document_store_save_path):
    if isinstance(document_store,ElasticsearchDocumentStore):
        return
    document_store.save(path)


def get_elastic_document_store():
    print('starting elastic docker')
    if not 'elasticsearch' in os.popen('docker ps').read():
        os.popen("""docker start fd2e31d49ed7f485d35f974594c404090269e20b9dc0ca9543d9c4a5bf626faf""")
        time.sleep(10)
        print(os.popen(
            """curl -XPUT -H "Content-Type: application/json" http://localhost:9200/_cluster/settings -d '{ "transient": { "cluster.routing.allocation.disk.threshold_enabled": false } }'"""))
        print(os.popen(
            """curl -XPUT -H "Content-Type: application/json" http://localhost:9200/_all/_settings -d '{"index.blocks.read_only_allow_delete": null}'"""))
        time.sleep(5)
    elastic_ds = ElasticsearchDocumentStore(host="localhost", username="", password="",
                                            index="document")
    return elastic_ds
