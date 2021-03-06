import json
import os
import time
from itertools import islice

from elasticsearch.helpers import bulk
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.retriever.base import BaseRetriever

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
    if isinstance(document_store, ElasticsearchDocumentStore):
        return
    document_store.save(path)


elastic_docker_id = """fd2e31d49ed7f485d35f974594c404090269e20b9dc0ca9543d9c4a5bf626faf"""


def get_elastic_document_store():
    def is_first_run():
        existing_images = os.popen('docker images').read()
        return 'elasticsearch' in existing_images and '7.9.2' in existing_images

    def create_image_and_volume():
        os.popen('mkdir -m777 -p elasticsearch/data')
        os.popen(
            'docker run -d -p 9200:9200 -e "discovery.type=single-node" -v $PWD/elasticsearch/data:/usr/share/elasticsearch/data --name elasticsearch elasticsearch:7.9.2')

    if is_first_run():
        create_image_and_volume()

    print('starting elastic docker')
    if 'elasticsearch' not in os.popen('docker ps').read():
        os.popen("""docker start %s""" % elastic_docker_id)
        time.sleep(25)
        print(os.popen(
            """curl -XPUT -H "Content-Type: application/json" http://localhost:9200/_cluster/settings -d '{ "transient": { "cluster.routing.allocation.disk.threshold_enabled": false } }'"""))
        print(os.popen(
            """curl -XPUT -H "Content-Type: application/json" http://localhost:9200/_all/_settings -d '{"index.blocks.read_only_allow_delete": null}'"""))
        time.sleep(5)
    elastic_ds = ElasticsearchDocumentStore(host="localhost", username="", password="",
                                            index="document", return_embedding=True)
    return elastic_ds


def update_elastic_embeddings(document_store: ElasticsearchDocumentStore, retriever: BaseRetriever,
                              update_existing=False):
    index = document_store.index

    result = document_store.get_all_documents_generator(index)
    for document_batch in get_batches_from_generator(result, 10_000):
        if len(document_batch) == 0:
            break
        if not update_existing:
            # take only documents with no embeddings
            document_batch = [d for d in document_batch if d.embedding is None]
        if len(document_batch) == 0:
            continue
        embeddings = retriever.embed_passages(document_batch)  # type: ignore
        assert len(document_batch) == len(embeddings)
        print('updating ',len(document_batch), ' embeddings')

        doc_updates = []
        for doc, emb in zip(document_batch, embeddings):
            update = {"_op_type": "update",
                      "_index": index,
                      "_id": doc.id,
                      "doc": {document_store.embedding_field: emb.tolist()},
                      }
            doc_updates.append(update)

        bulk(document_store.client, doc_updates, request_timeout=300, refresh=document_store.refresh_type)


def get_batches_from_generator(iterable, n):
    """
    Batch elements of an iterable into fixed-length chunks or blocks.
    """
    it = iter(iterable)
    x = tuple(islice(it, n))
    while x:
        yield x
        x = tuple(islice(it, n))
