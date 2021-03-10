import json

from dpr.scripts.utils import get_evidence_ids
from tqdm import tqdm

split = 'dev'
base_data_path = '../../data/strategyqa/'


def get_all_evidence_ids():
    train_json = '../../data/strategyqa/%s.json' % split
    with open(train_json, 'r', encoding="utf8") as train_file:
        questions = json.load(train_file)
        ids = sum([get_evidence_ids(q) for q in questions], [])
        return set(ids)


all_evidence_ids = get_all_evidence_ids()

startQA_corpus = base_data_path + 'corpus-enwiki-20200511-cirrussearch-parasv2.jsonl/enwiki-20200511-cirrussearch-parasv2.jsonl'

all = {}
count = 0
limit = 100
with open(startQA_corpus, 'r') as corpus:
    for line in tqdm(corpus):
        count += 1
        if count % 10 == 0:
            print('done ', count)
        if count > limit:
            break

        example = json.loads(line)
        title_ = example['title']
        id_ = example["para_id"]
        title__id_ = title_ + '-' + str(id_)
        if title__id_ not in all_evidence_ids:
            continue

        l = {'title': title_, 'content': example['para'], "para_index": id_}
        all[title__id_] = l

print('len all ', len(all))
with open(base_data_path + 'strategyqa_dev_paragraphs.json', 'w') as new_file:
    new_file.write(json.dumps(all))
