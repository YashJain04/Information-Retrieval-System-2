# imports
import numpy as np
from beir.reranking.rerank import Rerank

# custom reranker class to parse in a different way which is needed for some models such as BERT
class CustomRerank(Rerank):
    def rerank(self, corpus, query_dict, bm25_results, top_k=100):
        '''
        base rerank function used for both title and text
        '''
        results = {}
        for query_id, doc_scores in bm25_results.items():
            sentence_pairs = []
            doc_ids = []
            for doc_id, _ in doc_scores.items(): # iterate over dictionary items (doc_id and score)
                doc_text = corpus[doc_id]['title'] + " " + corpus[doc_id]['text'] # use both title and text
                sentence_pairs.append([query_dict[query_id], doc_text])
                doc_ids.append(doc_id)
            
            # get predictions from the cross encoder
            predictions = self.cross_encoder.predict(sentence_pairs, batch_size=self.batch_size)
            
            # convert each prediction to a scalar
            rerank_scores = []
            for pred in predictions:
                if isinstance(pred, np.ndarray):
                    if pred.size == 1:
                        score = pred.item()
                    else:
                        score = float(pred.flat[0])
                elif hasattr(pred, "item"):
                    score = pred.item()
                else:
                    score = float(pred)
                rerank_scores.append(score)
            
            # zip and sort the results based on the new scores
            reranked = sorted(zip(doc_ids, rerank_scores), key=lambda x: x[1], reverse=True)
            results[query_id] = reranked
        return results
    
    def rerank_head_only(self, corpus, query_dict, bm25_results, top_k=100):
        '''
        head/title only rerank function
        '''
        results = {}
        for query_id, doc_scores in bm25_results.items():
            sentence_pairs = []
            doc_ids = []
            for doc_id, _ in doc_scores.items(): # iterate over dictionary items (doc_id and score)
                doc_text = corpus[doc_id]['title'] # only use title
                sentence_pairs.append([query_dict[query_id], doc_text])
                doc_ids.append(doc_id)

            # get predictions from the cross encoder
            predictions = self.cross_encoder.predict(sentence_pairs, batch_size=self.batch_size)

            # convert each prediction to a scalar
            rerank_scores = []
            for pred in predictions:
                if isinstance(pred, np.ndarray):
                    if pred.size == 1:
                        score = pred.item()
                    else:
                        score = float(pred.flat[0])
                elif hasattr(pred, "item"):
                    score = pred.item()
                else:
                    score = float(pred)
                rerank_scores.append(score)

            # zip and sort the results based on the new scores
            reranked = sorted(zip(doc_ids, rerank_scores), key=lambda x: x[1], reverse=True)
            results[query_id] = reranked
        return results