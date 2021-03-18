import json
from dpr.scripts.utils import get_evidence_ids

split = 'train'
paragraphs_json = '../../data/strategyqa/strategyqa_%s_paragraphs.json' % split
train_json = '../../data/strategyqa/%s.json' % split
with open(paragraphs_json, 'r') as para_file:
    paragraphs = json.load(para_file)
with open(train_json, 'r', encoding="utf8") as train_file:
    questions = json.load(train_file)

data_to_write = []
for question in questions:
    evidence_ids = get_evidence_ids(question)
    positive_cntxs = []
    for id in evidence_ids:
        para_data = paragraphs[id]
        cntx = {'title': para_data['title'], 'text': para_data['content'], 'passage_id': id}
        positive_cntxs.append(cntx)

    data_to_write.append(
        {
            'dataset': 'startegyqa',
            'question': question['question'],
            'positive_ctxs': positive_cntxs,
            'negative_ctxs': [],
            'hard_negative_ctxs': [],
            'answers': [str(question['answer'])]
        }
    )

with open('../../data/strategyqa/%s_dpr.json' % split, 'w', encoding='utf-8') as f:
    json.dump(data_to_write, f)
