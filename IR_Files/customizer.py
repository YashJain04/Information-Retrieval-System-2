import numpy as np
from beir.reranking.rerank import Rerank

class CustomRerank(Rerank):
    def rerank(self, corpus, query_dict, bm25_results, top_k=100):
        results = {}
        for query_id, doc_scores in bm25_results.items():
            sentence_pairs = []
            doc_ids = []
            # Iterate over dictionary items (doc_id, score)
            for doc_id, _ in doc_scores.items():
                doc_text = corpus[doc_id]['title'] + " " + corpus[doc_id]['text']
                sentence_pairs.append([query_dict[query_id], doc_text])
                doc_ids.append(doc_id)
            
            # Get predictions from the cross encoder.
            predictions = self.cross_encoder.predict(sentence_pairs, batch_size=self.batch_size)
            
            # Convert each prediction to a scalar:
            rerank_scores = []
            for pred in predictions:
                # If pred is a numpy array and has more than one element, take the first element.
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
            
            # Zip and sort the results based on the new scores.
            reranked = sorted(zip(doc_ids, rerank_scores), key=lambda x: x[1], reverse=True)
            results[query_id] = reranked
        return results