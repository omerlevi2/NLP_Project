from collections import Counter

from dpr.retrievers.dataset.StrategyQADataset import StrategyQADataset


def should_skip(example):
    contexts = example['positive_ctxs']
    return len(contexts) < 2


def fix_contexts(contexts):
    if len(contexts) >= 9:
        return

    num_to_fill = 9 - len(contexts)
    for i in range(num_to_fill):
        contexts.append(contexts[i])


qa_dataset = StrategyQADataset()
dataset = qa_dataset.dev_set()

# c = Counter()
# for ex in dataset:
#     contexts = ex['positive_ctxs']
#     c[len(contexts)] += 1
#
# c = {k: c[k] for k in sorted(c)}
# print(c)
# print(len(dataset))


c = Counter()
dataset = [ex for ex in dataset if not should_skip(ex)]
for ex in dataset:
    contexts = ex['positive_ctxs']
    fix_contexts(contexts)
    contexts = ex['positive_ctxs']
    c[len(contexts)] += 1

c = {k: c[k] for k in sorted(c)}
print(c)
print(len(dataset))

qa_dataset._write_as_json('split_dev_dpr_fixed_context.json', dataset)
