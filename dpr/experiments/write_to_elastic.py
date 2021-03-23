from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
import concurrent.futures
import objgraph
import gc
import dpr.experiments.document_store as doc_store_utils

from dpr.retrievers.corpus.StrategyQAWikiCorpus import StrategyQAWikiCorpus

# docker run -d -p 9200:9200 -e "discovery.type=single-node" -v $PWD/elasticsearch/data:/usr/share/elasticsearch/data --name elasticsearch elasticsearch:7.9.2
# OR
# docker start fd2e31d49ed7f485d35f974594c404090269e20b9dc0ca9543d9c4a5bf626faf
# curl -XPUT -H "Content-Type: application/json" http://localhost:9200/_cluster/settings -d '{ "transient": { "cluster.routing.allocation.disk.threshold_enabled": false } }'
# curl -XPUT -H "Content-Type: application/json" http://localhost:9200/_all/_settings -d '{"index.blocks.read_only_allow_delete": null}'

# saving:
# docker cp elasticsearch:/usr/share/elasticsearch/data docker_es_save

# setup persistence:
# sudo mkdir -p $PWD/elasticsearch/data
# sudo chmod 777 -R $PWD/elaticsearch/data
elastic_ds = ElasticsearchDocumentStore(host="localhost", username="", password="",
                                        index="document")

num_existing_docs = elastic_ds.get_document_count()
print('num docs', num_existing_docs)
if num_existing_docs == 0:
    raise Exception("no elastic saved data!!")


def write_dicts(elastic, dicts):
    print('starting to write', len(dicts))
    elastic.write_documents(dicts,batch_size=batch_size)



import time


def get_dicts():
    i = 0
    for x in StrategyQAWikiCorpus().iter_json_batches(batch_size=batch_size, offset=num_existing_docs):
        i = i +1
        if i % 10 == 0:
            time.sleep(3)
        yield x


batch_size = 100_00
max_size = 9999999999999999
# dicts = []
futures = []
# for i, json in enumerate(StrategyQAWikiCorpus().iter_json_batches(batch_size=batch_size, offset=num_existing_docs)):
#     write_dicts(elastic_ds, json)
#
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(lambda dicts: write_dicts(elastic_ds, dicts),
                 get_dicts())
#
# # doc_store_utils.save_document_store(elastic_ds, 'elastic_db')

print('num docs', elastic_ds.get_document_count())
# print(elastic_ds.get_all_documents())
