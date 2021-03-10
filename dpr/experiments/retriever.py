from dataclasses import dataclass
from haystack.retriever.dense import DensePassageRetriever

batch_size: int = 1024
retriever_save_path = 'ret_save_file'


def get_retriever(document_store):
    query_model = "facebook/dpr-question_encoder-single-nq-base"
    passage_model = "facebook/dpr-ctx_encoder-single-nq-base"
    return DensePassageRetriever(
        document_store=document_store,
        query_embedding_model=query_model,
        passage_embedding_model=passage_model,
        max_seq_len_query=64,
        max_seq_len_passage=256,
        use_gpu=True,
        batch_size=batch_size,
        embed_title=True
    )


def save_retriever(retriever):
    print('saving retriever to ', retriever_save_path)
    retriever.save(retriever_save_path)


def load_retriever(document_store):
    return DensePassageRetriever.load(
        load_dir=retriever_save_path,
        document_store=document_store,
        max_seq_len_query=64,
        max_seq_len_passage=256,
        use_gpu=True,
        batch_size=batch_size,
        embed_title=True
    )


@dataclass
class RetrieverTrainParams:
    data_dir: str = '../../data/'
    save_dir: str = '/saved_models'

    train_filename: str = 'train_parsed.json'
    dev_filename: str = '../../data/dev_parsed.json'
    test_filename: str = dev_filename

    n_epochs: int = 1
    batch_size: int = 4
    grad_acc_steps: int = 4

    num_positives = 1
    num_hard_negatives = 1


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
