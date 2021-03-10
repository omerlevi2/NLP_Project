import json

from dpr.scripts.utils import get_evidence_ids


def get_all_evidence_ids():
    global split
    split = 'dev'
    train_json = '../../data/strategyqa/%s.json' % split
    with open(train_json, 'r', encoding="utf8") as train_file:
        questions = json.load(train_file)
        ids = sum([get_evidence_ids(q) for q in questions], [])
        return set(ids)


all_evidence_ids = get_all_evidence_ids()

