import json

from dpr.scripts.utils import get_evidence_ids

split = 'dev'

train_json = '../../data/strategyqa/%s.json' % split

with open(train_json, 'r', encoding="utf8") as train_file:
    questions = json.load(train_file)
    all_evidence_ids = set(sum([get_evidence_ids(q) for q in questions], []))
