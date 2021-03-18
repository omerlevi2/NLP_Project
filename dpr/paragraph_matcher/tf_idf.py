from dpr.paragraph_matcher.indexing_config import PATH_DOC_TERM_FREQ, PATH_PRE_FINAL_INDEX
from typing import List
from nltk.tokenize import word_tokenize 
from collections import Counter
import numpy as np
import ast
import json
import pickle
from tqdm import tqdm
from os import path
import os
from matplotlib import pyplot as plt

from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer
import nltk
# nltk.download("stopwords")
# nltk.download("punkt")
from string import punctuation, ascii_lowercase
from nltk.corpus import stopwords

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
allowed_symbols = set(l for l in ascii_lowercase)

PATH_NATURAL_QUES = "C:\\Users\\Omer\\Documents\\NLP class\\v1.0-simplified_simplified-nq-train.jsonl"
PATH_CORPUS_STRQA = "C:\\Users\\Omer\\Documents\\NLP class\\strategy_unzip\\corpus-enwiki-20200511-cirrussearch-parasv2.jsonl"
PATH_INV_IDX = 'dpr\paragraph_matcher\index\\inv_index_'
PATH_MAPPER = 'dpr\paragraph_matcher\index\mapper.pickle'
PATH_NUM_DOCS_PASSED = "dpr\paragraph_matcher\index\last_inv_idx_saved.txt"
PATH_DOC_FREQ = "dpr\paragraph_matcher\index\doc_freq"


def convert_to_passage(para: dict):
        start_token = para['annotations'][0]["long_answer"]["start_token"]
        end_token = para['annotations'][0]["long_answer"]["end_token"]
        passage = para["document_text"].split()[start_token:end_token]
        passage_to_text = ' '.join(passage)
        return passage_to_text

class TfIdf:
    def __init__(self):
        self.unigram_count =  Counter()
        self.document_term_frequency = Counter()
        self.word_document_frequency = {}
        self.inverted_index = {}
        self.doc_norms = {}
        self.n_docs = 0
        self.mapper = {}

    def update_counts_and_probabilities(self, sentence :List[str],document_id:int) -> None:
        sentence_len = len(sentence)
        self.document_term_frequency[document_id] = sentence_len
        for _,word in enumerate(sentence):
            self.unigram_count[word] += 1 
            if word not in self.inverted_index:
                self.inverted_index.update({word:{document_id:1}})
            if(document_id in self.inverted_index[word]):
                self.inverted_index[word][document_id] += 1
            else:
                    self.inverted_index[word].update({document_id: 1}) 
    
    def preprocess_sentence(self, sentence : str) -> List[str]:
        output_sentence = []
        for word in word_tokenize(sentence):
            word = word.lower()
            word = ''.join([i for i in word if i in allowed_symbols])
            if(word in stop_words):
                continue  
            word = stemmer.stem(word)
            if(len(word)>1):
                output_sentence.append(word)
        return output_sentence
    
    def load_index(self):
        self.inverted_index = pickle.load(open(PATH_PRE_FINAL_INDEX,'rb'))
        self.document_term_frequency = pickle.load(open(PATH_DOC_TERM_FREQ,'rb'))
            
        
    def fit(self,final_processing:bool) -> None:
        len_paras = []
        if(not final_processing):
            with open(PATH_CORPUS_STRQA + '\\enwiki-20200511-cirrussearch-parasv2.jsonl','r') as f:
                # x = json.load(f)
                # not_first_run = path.exists("dpr\paragraph_matcher\index\last_inv_idx_saved.txt")
                # docs_passed = self.get_last_run() if not_first_run else 0 #save a const var for later comparison
                # self.n_docs = docs_passed
                counter = 0
                for chunk in tqdm(f):
                    counter += 1
                    # if(counter < 14200989):continue
                    # if(counter > 35000100):
                    chunk = ast.literal_eval(chunk)
                    # print(chunk['para'])
                    # print('------------------------------------------------------------------------------------------------')
            #         if(counter<=self.n_docs): continue
            #         if((counter-1 == docs_passed) and docs_passed != 0):
            #             self.mapper = pickle.load(open(PATH_MAPPER,'rb'))
            #             self.inverted_index = pickle.load(open(PATH_INV_IDX,'rb'))
                    self.n_docs += 1
                    
                    # chunk = ast.literal_eval(chunk)
                    # para = self.preprocess_sentence(chunk['para'])
                    len_paras.append(len(chunk['para']))
                    
                    if(counter==1000000000):
                        plt.xlim([min(len_paras), max(len_paras)])
                        bins = np.arange(0, 10000, 5)
                        plt.hist(len_paras, bins=bins)
                        plt.show()
                        print('')
            #         self.mapper[self.n_docs] = (chunk['docid'],chunk['para_id'])
            #         # if not isinstance(para, str):
            #         #     continue
            #         if para:
            #             self.update_counts_and_probabilities(para,self.n_docs)
            # #             # backup every 5 million iterations
            #             if(self.n_docs % (5*(10**6)) == 0):
            #                 self.back_up_idx()
            # file_prefix = str(int(self.n_docs/(10**6)))
            # self.save_inv_idx(file_p5refix)
            # self.save_last_doc_saved()
            # self.save_mapper()
            # self.save_doc_term_freq()
            
        else:
            self.inverted_index = pickle.load(open('dpr\paragraph_matcher\index\inv_index_36','rb'))
            self.document_term_frequency = pickle.load(open(PATH_DOC_TERM_FREQ,'rb'))
            self.n_docs = 36617357
            self.compute_word_document_frequency()
            self.update_inverted_index_with_tf_idf_and_compute_document_norm()
            self.save_inv_idx('final_36_inv_idx')
            pickle.dump(self.word_document_frequency, open('dpr\paragraph_matcher\index\word_doc_freq','wb'))
            pickle.dump(self.doc_norms, open('dpr\paragraph_matcher\index\doc_norms','wb'))

    def compute_word_document_frequency(self):
        for word in self.inverted_index.keys():
            self.word_document_frequency[word] = len(self.inverted_index[word])
            
            
    def update_inverted_index_with_tf_idf_and_compute_document_norm(self):
        for term in self.inverted_index:
            for doc, freq in self.inverted_index[term].items():
                self.inverted_index[term][doc] = (freq / self.document_term_frequency[doc] * np.log10(self.n_docs/self.word_document_frequency[term]))
                if doc not in self.doc_norms:
                    self.doc_norms.update({doc:0}) 
                self.doc_norms[doc] += (self.inverted_index[term][doc]**2)
        print('g')
        for doc in self.doc_norms.keys():
            self.doc_norms[doc] = np.sqrt(self.doc_norms[doc]) 

    def save_inv_idx(self,file_prefix):
        with open(PATH_INV_IDX + file_prefix,'wb') as out:
            pickle.dump(self.inverted_index,out)
    
    def save_last_doc_saved(self):
        f = open(PATH_NUM_DOCS_PASSED,'w')
        as_str = str(self.n_docs)
        f.write(as_str)
        f.close()
    
    def get_last_run(self):
        f = open(PATH_NUM_DOCS_PASSED,'r')
        last_run = int(f.read())
        return last_run
    
    def save_mapper(self):
        with open(PATH_MAPPER,'wb') as out:
            pickle.dump(self.mapper,out)

    def save_doc_term_freq(self):
        with open(PATH_DOC_FREQ,'wb') as out:
                pickle.dump(self.document_term_frequency,out)
    
    def back_up_idx(self):
        os.makedirs('dpr\paragraph_matcher\index', exist_ok=True)
        file_prefix = str(int(self.n_docs/(10**6)))
        self.save_inv_idx(file_prefix)
        self.save_last_doc_saved()
        self.save_mapper()
        self.save_doct_term_freq()
        self.inverted_index = {}



if __name__ == '__main__':

    tf_idf = TfIdf()
    tf_idf.fit(final_processing=False)