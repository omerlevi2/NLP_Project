import json
from typing import List, Dict

paragraphs_json = '../data/strategyqa/strategyqa_train_paragraphs.json'
train_json = '../data/strategyqa/train.json'
with open(paragraphs_json, 'r') as para_file:
    paragraphs = json.load(para_file)
with open(train_json, 'r') as train_file:
    questions = json.load(train_file)


def get_evidence(startqa_example):
    def get_evidence_strings(evidence):
        if isinstance(evidence, str):
            return [evidence]
        ids = []
        for evi in evidence:
            ids.extend(get_evidence_strings(evi))
        return ids

    evidence = startqa_example['evidence']
    evidence_ids = get_evidence_strings(evidence)
    return [x for x in evidence_ids if not x == 'operation' and not x == 'no_evidence']


print(paragraphs)


def get_evidence(inputs):
    pass