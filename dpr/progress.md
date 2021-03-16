todo:
0)find out where was the retriver saved to.. probably /saved_models.. if I did run this shit

1) load trained retriever
3) update embeddings for dataset
4) load trained retriver and updated dataset
5) see it yield different results from untrained retriver

what is this? DPR model loaded from ret_save_file

how many epochs should dpr be trained for?

train retriver on startegyqa dataset

fix dev set paragraphs:
join_startegyqa_evidence.py does not work because there is no strategyqa_dev_paragraphs.json

# this should be a larger number.see comment on notion on uneven number of positives
num_positives:int = 1

deal with hard_negative_ctxs

done:
reformat files strategyQA files to include actual evidence moved all our stuff to dpr package
