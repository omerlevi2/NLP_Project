import json
import os
import sys

cwd = os.getcwd()
print('cwd is: ', cwd)
sys.path.append(cwd[:cwd.index('pycharm_project_') + + len('pycharm_project_') + 4])



doc_dir = "data/nq/"

startQA_corpus = doc_dir + "dev/" + 'corpus-enwiki-20200511-cirrussearch-parasv2.jsonl'
formated_file_name=doc_dir + 'dev/' + 'startqa_corpus_formatted_for_documentstore.json'

with open(startQA_corpus, 'r') as corpus:
    with open(formated_file_name, 'w+') as formatted_corpus:
        formatted_corpus.write('[\n')
        for line in corpus:
            example = json.loads(line)
            l = {'text': example['para'], 'meta': {'title': example['title']}}
            formatted_corpus.write(json.dumps(l) + '\n')
        formatted_corpus.write('\n]')
