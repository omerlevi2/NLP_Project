import json


def get_evidence_ids(startqa_example):
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


paragraphs_json = '../data/strategyqa/strategyqa_train_paragraphs.json'
train_json = '../data/strategyqa/train.json'
with open(paragraphs_json, 'r') as para_file:
    paragraphs = json.load(para_file)
with open(train_json, 'r', encoding="utf8") as train_file:
    questions = json.load(train_file)

for question in questions:
    evidence_ids = get_evidence_ids(question)
    positive_cntxs = []
    for id in evidence_ids:
        para_data = paragraphs[id]
        cntx = {'title': para_data['title'], 'text': para_data['content']}
        positive_cntxs.append(cntx)

    question['positive_ctxs'] = positive_cntxs

print(str(questions)[:1000])

with open('../../data/strategyqa/train_parsed.json', 'w', encoding='utf-8') as f: f.write(str(questions))

