from dataclasses import dataclass
import json


@dataclass
class StrategyQADataset:
    data_dir: str = 'data/strategyqa'
    train_filename: str = 'train_dpr.json'
    dev_filename: str = 'dev_dpr.json'
    test_filename: str = None

    def dev_set(self):
        with open(self.data_dir + '/' + self.dev_filename, 'r', encoding="utf8") as train_file:
            return json.load(train_file)
