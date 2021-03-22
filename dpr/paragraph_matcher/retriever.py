from dpr.paragraph_matcher.tf_idf import TfIdf
from typing import List,Dict
from collections import Counter
import numpy as np



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
    tf_idf = TfIdf()
    tf_idf.load_index()

