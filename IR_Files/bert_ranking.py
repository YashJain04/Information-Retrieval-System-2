import math
from sentence_transformers import SentenceTransformer, util

def normalize_scores(ranked_docs):
    """
    Normalize scores to [0,1] range, similar to the approach in ranking.py
    """
    if not ranked_docs:
        return []
    max_score = max(score for _, score in ranked_docs)
    min_score = min(score for _, score in ranked_docs)
    if max_score == min_score:
        return [(doc_id, 1.0) for doc_id, _ in ranked_docs]
    return [(doc_id, (score - min_score) / (max_score - min_score)) for doc_id, score in ranked_docs]


class BERT_Ranker:
    """
    A BERT-based ranker that embeds all documents once (using SentenceTransformers),
    and computes cosine similarity for each query.
    """

    def __init__(self, docs_dict, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        docs_dict: a dict {doc_id -> text_string} for all documents
        model_name: which SentenceTransformer checkpoint to use
        """
        self.model = SentenceTransformer(model_name)
        self.docs_dict = docs_dict

        # Precompute embeddings for every document
        self.doc_embeddings = {}
        doc_ids = list(docs_dict.keys())
        doc_texts = [docs_dict[d] for d in doc_ids]

        # Encode all documents at once
        embeddings = self.model.encode(doc_texts, convert_to_tensor=True)

        # Store in a dictionary {doc_id -> embedding_tensor}
        for i, doc_id in enumerate(doc_ids):
            self.doc_embeddings[doc_id] = embeddings[i]

    def rank_documents(self, query_terms):
        """
        Given a list of preprocessed query tokens, join them into a string,
        embed the query, then compute cosine similarity with each document embedding.
        Returns a list of (doc_id, score) sorted in descending order of similarity.
        """
        # Join the tokenized query into a single string
        query_text = " ".join(query_terms)
        # Encode the query (shape: [dim])
        query_emb = self.model.encode([query_text], convert_to_tensor=True)[0]

        # Calculate cosine similarity with each document
        scores = {}
        for doc_id, doc_emb in self.doc_embeddings.items():
            cos_score = util.cos_sim(query_emb, doc_emb).item()
            scores[doc_id] = cos_score

        # Sort results by score in descending order
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs
