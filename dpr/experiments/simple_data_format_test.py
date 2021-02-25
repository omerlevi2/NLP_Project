import json
import os
import time
import sys

cwd = os.getcwd()
print('cwd is: ', cwd)
sys.path.append(cwd[:cwd.index('pycharm_project_') + + len('pycharm_project_') + 4])

from utils import compute
torch = compute.get_torch()
from haystack import Finder
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http, eval_data_from_json,squad_json_to_jsonl,eval_data_from_jsonl
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.retriever.dense import DensePassageRetriever
from haystack.document_store.faiss import FAISSDocumentStore


doc_dir = "data/nq/"
query_model = "facebook/dpr-question_encoder-single-nq-base"
passage_model = "facebook/dpr-ctx_encoder-single-nq-base"
save_dir = "../saved_models/dpr"
startQA_corpus = doc_dir + "dev/" + 'corpus-enwiki-20200511-cirrussearch-parasv2.jsonl'
formated_file_name=doc_dir + 'dev/' + 'startqa_corpus_formatted_for_documentstore.json'

dicts = []
with open(startQA_corpus, 'r') as corpus:
    with open(formated_file_name, 'w+') as formatted_corpus:
        formatted_corpus.write('[\n')
        for line in corpus:
            example = json.loads(line)
            l = {'text': example['para'], 'meta': {'title': example['title']}}
            formatted_corpus.write(json.dumps(l) + '\n')
        formatted_corpus.write('\n]')

document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
# dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
# Now, let's write the dicts containing documents to our DB.
document_store.write_documents(dicts)


retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model=query_model,
    passage_embedding_model=passage_model,
    max_seq_len_query=64,
    max_seq_len_passage=256,
    # use_gpu=False
)

document_store.update_embeddings(retriever)

print(3)