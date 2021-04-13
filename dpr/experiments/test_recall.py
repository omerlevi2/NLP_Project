import json

from dpr.retrievers.dataset.StrategyQADataset import StrategyQADataset

with open('strategy_retrieved_passages_10.json', 'r', encoding="utf8") as file:
    # with open('strategy_retrieved_passages.json', 'r', encoding="utf8") as file:
    data = json.load(file)


def get_passages_from_value(value):
    return [d['text'] for d in value]


q_to_retrieved = {q: get_passages_from_value(v) for q, v in data.items()}

qa_dataset = StrategyQADataset()
qa_dataset_dev = qa_dataset.dev_set()


def extract_key(entry):
    return entry['question']


def extract_passages(entry):
    cntx = entry['positive_ctxs']
    return get_passages_from_value(cntx)


q_to_gold = {extract_key(e): extract_passages(e) for e in qa_dataset_dev}

for key in q_to_gold:
    assert key in q_to_retrieved


def recall(relevant_paragraphs, retrieved_paragraphs):
    return len(set(relevant_paragraphs).intersection(retrieved_paragraphs)) / \
           len(relevant_paragraphs)


# print([len(v) for v in q_to_retrieved.values()])


def avg_recall_at(at=10):
    recalls = [recall(q_to_gold[question], q_to_retrieved[question][:at]) for question in q_to_gold]
    return sum(recalls) / len(recalls)


# recalls = [recall(q_to_gold[question], q_to_retrieved[question]) for question in q_to_gold]
for i in range(1, 11):
    print(' recall at', i, avg_recall_at(i))


def get_recalls_in_range(start, end):
    recalls = [(question, recall(q_to_gold[question], q_to_retrieved[question]))
               for question in q_to_gold]
    return [(x, y) for x, y in recalls if start <= y <= end]


print(get_recalls_in_range(0.,0.1))
print('\n\n')
print(get_recalls_in_range(0.6,1.))
