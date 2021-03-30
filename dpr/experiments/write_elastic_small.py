from dpr.retrievers.dataset.StrategyQADataset import StrategyQADataset


def get_all_positive_contexts(dataset):
    gold_paragraphs = set()
    for example in dataset:
        paragraphs = [x['text'] for x in example['positive_ctxs']]
        gold_paragraphs.update(paragraphs)
    return gold_paragraphs


train_para = get_all_positive_contexts(StrategyQADataset().train_set())
dev_para = get_all_positive_contexts(StrategyQADataset().dev_set())
all_paras = train_para.union(dev_para)

print(len(dev_para) + len(train_para))
print(len(all_paras))
