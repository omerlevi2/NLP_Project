import ast
from nltk.tokenize import word_tokenize 
from tqdm import tqdm
import numpy as np
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download("stopwords")
nltk.download("punkt")
from string import punctuation, ascii_lowercase
from nltk.corpus import stopwords
from typing import List,Dict
import pickle


PATH_CORPUS_STRQA = "C:\\Users\\Omer\\Documents\\NLP class\\strategy_unzip\\corpus-enwiki-20200511-cirrussearch-parasv2.jsonl"

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
allowed_symbols = set(l for l in ascii_lowercase)

def fit(self) -> None:
    with open(PATH_CORPUS_STRQA + '\\enwiki-20200511-cirrussearch-parasv2.jsonl','r') as f:
        word_mapper = {}
        unique = 0
        for chunk in tqdm(f):
            chunk = ast.literal_eval(chunk)
            para = word_tokenize(chunk['para'])
            for word in para:
                if(word not in word_mapper):
                    unique += 1
                    word_mapper[word] = unique
    
    
def preprocess_sentence(sentence : str) -> List[str]:
    output_sentence = []
    for word in word_tokenize(sentence):
        word = word.lower()
        word = ''.join([i for i in word if i in allowed_symbols])
        if(word in stop_words):
            continue  
        word = stemmer.stem(word)
        if(len(word)>1):
            output_sentence.append(word)
        
        
        
        ### END YOUR CODE
    return output_sentence

if __name__ == '__main__':
    with open(PATH_CORPUS_STRQA + '\\enwiki-20200511-cirrussearch-parasv2.jsonl','r') as f:
        word_mapper = {}
        unique = np.int16(0)
        for chunk in tqdm(f):
            chunk = ast.literal_eval(chunk)
            # para = word_tokenize(chunk['para'])
            para = preprocess_sentence(chunk['para'])
            for word in para:
                if(word not in word_mapper):
                    unique  = np.int16(unique+1)
                    word_mapper[word] = unique
        pickle.dump(word_mapper,open('\index\word_mapper.pickle'),'wb')