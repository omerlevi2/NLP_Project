from haystack.retriever.dense import DensePassageRetriever

batch_size: int = 16


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
        batch_size=batch_size
    )
