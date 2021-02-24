import json
from typing import List, Dict

paragraphs_json = '../data/strategyqa/strategyqa_train_paragraphs.json'
train_json = '../data/strategyqa/train.json'
with open(paragraphs_json, 'r') as para_file:
    paragraphs = json.load(para_file)
with open(train_json, 'r') as train_file:
    questions = json.load(train_file)


def get_evidence(startqa_example):
    def get_evidence_ids(evidence):
        if isinstance(evidence, str):
            if evidence == 'operation' or evidence == 'no_evidence':
                return []
            else:
                return [evidence]
        ids = []
        for evi in evidence:
            ids.extend(get_evidence_ids(evi))
        return ids

print(paragraphs)


def get_evidence(inputs):
    pass
