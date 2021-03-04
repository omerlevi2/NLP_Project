from typing import List,Dict
from nltk.tokenize import word_tokenize 
from collections import Counter
import numpy as np
import ast
import json
import pickle
from tqdm import tqdm
from os import path
import os

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
PATH_INV_IDX = 'dpr\paragraph_matcher\index\inv_index.pickle'
PATH_MAPPER = 'dpr\paragraph_matcher\index\mapper.pickle'
PATH_NUM_DOCS_PASSED = "dpr\paragraph_matcher\index\last_inv_idx_saved.txt"


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
        # self.sentence_preprocesser = preprocess_sentence
        self.mapper = {}
        # self.bow_path = BOW_PATH

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
            
        
    def fit(self) -> None:
        with open(PATH_CORPUS_STRQA + '\\enwiki-20200511-cirrussearch-parasv2.jsonl','r') as f:
            not_first_run = path.exists("dpr\paragraph_matcher\index\last_inv_idx_saved.txt")
            docs_passed = self.get_last_run() if not_first_run else 0 #save a const var for later comparison
            self.n_docs = docs_passed
            counter = 0
            for chunk in tqdm(f):
                counter += 1
                if(counter<=self.n_docs): continue
                if((counter-1 == docs_passed) and docs_passed != 0):
                    self.mapper = pickle.load(open(PATH_MAPPER,'rb'))
                    self.inverted_index = pickle.load(open(PATH_INV_IDX,'rb'))
                self.n_docs += 1
                chunk = ast.literal_eval(chunk)
                # para = word_tokenize(chunk['para'])
                para = self.preprocess_sentence(chunk['para'])
                # print(para)
                # print(len(para))
                # for sentence in chunck:
                self.mapper[self.n_docs] = (chunk['docid'],chunk['para_id'])
                # if not isinstance(para, str):
                #     continue
                # sentence = self.sentence_preprocesser(sentence)
                if para:
                    self.update_counts_and_probabilities(para,self.n_docs)
                    # backup every 5 million iterations
                    if(self.n_docs % (5*(10**6)) == 0):
                        os.makedirs('dpr\paragraph_matcher\index', exist_ok=True)
                        self.save_inv_idx()
                        self.save_last_doc_saved()
                        self.save_mapper()
        self.save_inv_idx()
        self.save_last_doc_saved()
        self.save_mapper()
        # self.compute_word_document_frequency()
        # self.update_inverted_index_with_tf_idf_and_compute_document_norm()
             
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
        
        for doc in self.doc_norms.keys():
            self.doc_norms[doc] = np.sqrt(self.doc_norms[doc]) 

    def save_inv_idx(self):
        with open(PATH_INV_IDX,'wb') as out:
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

class DocumentRetriever:
    def __init__(self, tf_idf):
        self.sentence_preprocesser = self.preprocess_sentence
        self.vocab = set(tf_idf.unigram_count.keys())
        self.n_docs = tf_idf.n_docs
        self.inverted_index = tf_idf.inverted_index
        self.word_document_frequency = tf_idf.word_document_frequency
        self.doc_norms = tf_idf.doc_norms

    def rank(self, query: Dict[str, int], documents: Dict[str, Counter], metric: str) -> Dict[str, float]:
        result = {}  # key: DocID , value : float , simmilarity to query
        query_len = np.sum(np.array(list(query.values())))
        for term, count in query.items(): #in this loop we're updating the query's weights 
            query[term] = (count / query_len * np.log10(tf_idf.n_docs / tf_idf.word_document_frequency[term]))
            for doc, freq in documents[term].items():
                if (doc not in result):
                    result.update({doc: 0})
                if metric == 'inner_product':
                    result[doc] += query[term] * freq
            
                if metric == 'cosine':
            
                    result[doc] += (query[term] * freq / self.doc_norms[doc])

                
        return result

    def sort_and_retrieve_k_best(self, scores: Dict[str, float], k: int):
        return list({k: v for k, v in sorted(scores.items(), key=lambda item: item[1],reverse=True)})[:k]
        

    def reduce_query_to_counts(self, query: List) :
        return dict(Counter(query)) # rank get Dict as input so we used this cast (even that a counter is a dict)
        

    def get_top_k_documents(self, query: str, metric: str, k=5) -> List[str]:
        query = self.sentence_preprocesser(query)
        query = [word for word in query if word in self.vocab]  # filter nan
        query_bow = self.reduce_query_to_counts(query)
        relavant_documents = {word: self.inverted_index.get(word) for word in query}
        ducuments_with_similarity = self.rank(query_bow, relavant_documents, metric)
        return self.sort_and_retrieve_k_best(ducuments_with_similarity, k)
        

if __name__ == '__main__':
    # path = PATH_N_QUES
    # with open(path + '\\simplified-nq-train.jsonl','rb') as f:
    #     for line in f: 
    #         long_passage = convert_to_passage(line)
    #         if(len(long_passage)==0):
    #             continue
    #         print(long_passage)

    tf_idf = TfIdf()
    tf_idf.fit()