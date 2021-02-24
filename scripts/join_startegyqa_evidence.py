import json

paragraphs_json = '../data/strategyqa/strategyqa_train_paragraphs.json'
train_json = '../data/strategyqa/train.json'
with open(paragraphs_json, 'r') as para_file:
    paragraphs = json.load(para_file)
with open(train_json, 'r') as train_file:
    questions = json.load(train_file)

print(paragraphs)


def get_evidence(inputs):
    pass
