import json

with open('../../data/strategyqa/train_dpr.json', 'r') as f:
    train_dpr = json.load(f)
new_train_list = []
new_dev_list = []
for i in range(len(train_dpr)):
    if i <= (len(train_dpr) // 10):
        new_dev_list.append(train_dpr[i])
    else:
        new_train_list.append(train_dpr[i])

with open('../../data/strategyqa/train_dpr_strategyqa/split_train_dpr.json', 'w') as f:
    json.dump(new_train_list, f)

with open('../../data/strategyqa/train_dpr_strategyqa/split_dev_dpr.json', 'w') as f:
    json.dump(new_dev_list, f)
