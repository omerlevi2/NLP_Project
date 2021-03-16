from haystack.document_store.memory import InMemoryDocumentStore
from haystack.retriever.dense import DensePassageRetriever

batch_size: int = 1024
retriever_save_path = 'ret_save_file'


def get_retriever(document_store,
                  query_model="facebook/dpr-question_encoder-single-nq-base",
                  passage_model="facebook/dpr-ctx_encoder-single-nq-base"):
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


def get_retriever_for_training(query_model="facebook/dpr-question_encoder-single-nq-base",
                               passage_model="facebook/dpr-ctx_encoder-single-nq-base"):
    return get_retriever(InMemoryDocumentStore(), query_model=query_model, passage_model=passage_model)


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
