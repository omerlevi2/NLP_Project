todo:
1) create adjusted natural questions:
   * drop negative contexts
     * check if old nq or corpus is in ../../data so I can delete it and clear some space 
   * print to file in the same format

2) evaluate metrics properly



test if training works

figure out how much to put in positve_context..maybe can just use full and it will work.. if not need to just buff up not down

1) load trained retriever 3) update embeddings for dataset 4) load trained retriver and updated dataset 5) see it yield
   different results from untrained retriver
   
how many epochs should dpr be trained for? yes:its fine tuning so less than 100 no:--100 on small dataset--

# this should be a larger number.see comment on notion on uneven number of positives

plug into existing ir pipeline:
see paragraphs = { None: lambda **kwargs: None,
"IR-Q": self._ir_q,
"ORA-P": self._ora_p,
"IR-ORA-D": self._ir_ora_d,
"IR-D": self._ir_d, }[self._paragraphs_source](**kwargs)

done:
reformat files strategyQA files to include actual evidence moved all our stuff to dpr package
hard_negatives: looks like not that important for nq