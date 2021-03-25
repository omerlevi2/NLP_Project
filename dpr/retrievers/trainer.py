from dataclasses import dataclass

from dpr.retrievers.retrieves import retriever_save_path, save_retriever


@dataclass
class RetrieverTrainParams:
    save_dir: str = retriever_save_path

    n_epochs: int = 1
    batch_size: int = 8
    grad_acc_steps: int = 2

    # todo should change
    num_positives: int = 1
    # todo should resolve
    num_hard_negatives: int = 1


def train(retriever, dataset, params: RetrieverTrainParams, save=True):
    retriever.train(
        data_dir=dataset.data_dir,
        train_filename=dataset.train_filename,
        dev_filename=dataset.dev_filename,
        n_epochs=params.n_epochs,
        batch_size=params.batch_size,
        grad_acc_steps=params.grad_acc_steps,
        save_dir=params.save_dir,
        evaluate_every=500,
        embed_title=True,
        num_positives=params.num_positives,
        num_hard_negatives=params.num_hard_negatives
    )
    if save:
        save_retriever(retriever, params.save_dir)
