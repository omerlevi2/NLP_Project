from typing import List,Dict
from nltk.tokenize import word_tokenize 
from collections import Counter
import numpy as np
import ast
import json


def preprocess_sentence(sentence : str) -> List[str]:
    output_sentence = []
    for word in word_tokenize(sentence):
        output_sentence.append(word)

    return output_sentence

def convert_to_passage(line):
        data_dict = json.loads(line.decode('utf-8'))
        start_token = data_dict['annotations'][0]["long_answer"]["start_token"]
        end_token = data_dict['annotations'][0]["long_answer"]["end_token"]
        passage = data_dict["document_text"].split()[start_token:end_token]
        passage_to_text = ' '.join(passage)
        return passage_to_text

class TfIdf:
    def __init__(self):
        self.unigram_count =  Counter()
        self.document_term_frequency = Counter()
        self.word_document_frequency = {}
        self.inverted_index = {}
        self.doc_norms = {}
        self.n_docs = -1
        self.sentence_preprocesser = preprocess_sentence
        # self.bow_path = BOW_PATH

    def update_counts_and_probabilities(self, sentence :List[str],document_id:int) -> None:
        sentence_len = len(sentence)
        self.document_term_frequency[document_id] = sentence_len
        for i,word in enumerate(sentence):
            ### YOUR CODE HERE
            self.unigram_count[word] += 1 
            if word not in self.inverted_index:
                self.inverted_index.update({word:{document_id:1}})
            if(document_id in self.inverted_index[word]):
                self.inverted_index[word][document_id] += 1
            else:
                    self.inverted_index[word].update({document_id: 1}) 
            ### END YOUR CODE
        
    def fit(self) -> None:
        path = "C:\\Users\\Omer\\Documents\\NLP class\\strategy_unzip\\corpus-enwiki-20200511-cirrussearch-parasv2.jsonl"
        with open(path + '\\enwiki-20200511-cirrussearch-parasv2.jsonl','r') as f:
            for line in f:
                line = ast.literal_eval(line)
                para = word_tokenize(line['para'])
                # for sentence in chunck:
                self.n_docs += 1 
                # if not isinstance(para, str):
                #     continue
                # sentence = self.sentence_preprocesser(sentence)
                if para:
                    self.update_counts_and_probabilities(para,self.n_docs)
#         self.save_bow() # bow is 'bag of words'
        self.compute_word_document_frequency()
        self.update_inverted_index_with_tf_idf_and_compute_document_norm()
             
    def compute_word_document_frequency(self):
        for word in self.inverted_index.keys():
            ### YOUR CODE HERE
            self.word_document_frequency[word] = len(self.inverted_index[word])
            ### END YOUR CODE
            
    def update_inverted_index_with_tf_idf_and_compute_document_norm(self):
        ### YOUR CODE HERE
        for term in self.inverted_index:
            for doc, freq in self.inverted_index[term].items():
                self.inverted_index[term][doc] = (freq / self.document_term_frequency[doc] * np.log10(self.n_docs/self.word_document_frequency[term]))
                if doc not in self.doc_norms:
                    self.doc_norms.update({doc:0}) 
                self.doc_norms[doc] += (self.inverted_index[term][doc]**2)
        ### END YOUR CODE
        for doc in self.doc_norms.keys():
            self.doc_norms[doc] = np.sqrt(self.doc_norms[doc]) 
    
    

if __name__ == '__main__':
    path = "C:\\Users\\Omer\\Documents\\NLP class\\v1.0-simplified_simplified-nq-train.jsonl"
    with open(path + '\\simplified-nq-train.jsonl','rb') as f:
        for line in f: 
            long_passage = convert_to_passage(line)
            if(len(long_passage)==0):
                continue
            print(long_passage)

    tf_idf = TfIdf()
    tf_idf.fit()