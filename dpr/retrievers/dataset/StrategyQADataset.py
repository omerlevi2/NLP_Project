from dataclasses import dataclass
import json


@dataclass
class StrategyQADataset:
    data_dir: str = 'data/strategyqa'
    # data_dir: str = 'data/train_dpr_strategyqa'
    # train_filename: str = 'split_train_dpr.json'
    train_filename: str = 'train_dpr.json'
    # train_filename: str = 'split_train_dpr_fixed_context.json'
    # dev_filename: str = 'split_dev_dpr.json'
    dev_filename: str = 'dev_dpr.json'
    # dev_filename: str = 'split_dev_dpr_fixed_context.json'
    test_filename: str = None

    def dev_set(self):
        return self._get_as_json(self.dev_filename)

    def train_set(self):
        return self._get_as_json(self.train_filename)

    # format is dataset": "startegyqa", "question": "Are more people today related to Genghis Khan than Julius Caesar?", "positive_ctxs":[]..
    def _get_as_json(self, filename):
        with open(self.data_dir + '/' + filename, 'r', encoding="utf8") as train_file:
            return json.load(train_file)

    def _write_as_json(self, new_filename, list_of_jsons):
        with open(self.data_dir + '/' + new_filename, 'w', encoding="utf8") as file:
            return json.dump(list_of_jsons, file)
