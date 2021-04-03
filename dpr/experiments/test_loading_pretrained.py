import document_store
import dpr.retrievers.retrieves as retrieves

ds = document_store.get_elastic_document_store()
retriever = retrieves.load_retriever(ds, 'ret_save_file_nq_only')

#USE THIS IF you dont want to index every time
# update_document_store_embeddings_and_save(ds,retriever)
ds.update_embeddings(retriever)

res = retriever.retrieve('"who sings does he love me with reba', top_k=3)

print(res)
