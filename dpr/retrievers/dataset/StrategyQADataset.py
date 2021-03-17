from dataclasses import dataclass


@dataclass
class StrategyQADataset:
    data_dir: str = 'data/strategyqa'
    train_filename: str = 'train_dpr.json'
    dev_filename: str = 'dev_dpr.json'
    test_filename: str = None
