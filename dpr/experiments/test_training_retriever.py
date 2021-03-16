import dpr.retrievers.retrieves as retrieves
import dpr.retrievers.trainer as trainer

load = True

doc_dir = 'data/'
formated_file_name = doc_dir + 'startqa_corpus_formatted_for_documentstore.json'
# formated_file_name = doc_dir + 'sample_startqa_corpus_formatted_for_documentstore.jsonl'

if load:
    retriever = retrieves.load_retriever()
else:
    retriever = retrieves.get_retriever_for_training()
    trainer.train(retriever,
                  trainer.RetrieverTrainParams(num_hard_negatives=0, n_epochs=1))

print(3)
