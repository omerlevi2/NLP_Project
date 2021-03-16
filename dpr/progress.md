todo:
0)find out where was the retriver saved to.. probably /saved_models.. if I did run this shit

1) load trained retriever
3) update embeddings for dataset
4) load trained retriver and updated dataset
5) see it yield different results from untrained retriver

what is this? DPR model loaded from ret_save_file

how many epochs should dpr be trained for?


# this should be a larger number.see comment on notion on uneven number of positives
num_positives:int = 1
deal with hard_negative_ctxs... find a reasonable number

plug into existing ir pipeline:
see paragraphs = {
            None: lambda **kwargs: None,
            "IR-Q": self._ir_q,
            "ORA-P": self._ora_p,
            "IR-ORA-D": self._ir_ora_d,
            "IR-D": self._ir_d,
        }[self._paragraphs_source](**kwargs)


done:
reformat files strategyQA files to include actual evidence moved all our stuff to dpr package
