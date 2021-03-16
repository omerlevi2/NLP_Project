from dataclasses import dataclass

from dpr.retrievers.retrieves import retriever_save_path


@dataclass
class RetrieverTrainParams:
    data_dir: str = '../../data/strategyqa'
    save_dir: str = 'saved_models'

    train_filename: str = 'train_dpr.json'
    dev_filename: str = 'dev_dpr.json'
    test_filename: str = dev_filename

    n_epochs: int = 1
    batch_size: int = 8
    grad_acc_steps: int = 2

    # todo should change
    num_positives: int = 1
    # todo should resolve
    num_hard_negatives: int = 1


def train(retriever, params: RetrieverTrainParams):
    retriever.train(
        data_dir=params.data_dir,
        train_filename=params.train_filename,
        dev_filename=params.dev_filename,
        test_filename=params.dev_filename,
        n_epochs=params.n_epochs,
        batch_size=params.batch_size,
        grad_acc_steps=params.grad_acc_steps,
        save_dir=params.save_dir,
        evaluate_every=3000,
        embed_title=True,
        num_positives=params.num_positives,
        num_hard_negatives=params.num_hard_negatives
    )

    def save_retriever(retriever):
        print('saving retriever to ', retriever_save_path)
        retriever.save(retriever_save_path)