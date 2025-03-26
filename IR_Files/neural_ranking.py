import tensorflow as tf
import json
import torch
from ranking import BM25
from customizer import CustomRerank
from utils import *
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank

def create_model(model_type, model_name, documents, inverted_index, documents_length):
    '''
    Load a specific model in [BM25, BERT, ELECTRA]
    '''
    if model_type == 'BM25':
        return BM25(inverted_index, documents_length)
    
    elif model_type == 'cross-encoder':
        return CrossEncoder(model_name)
    
    else:
        raise ValueError(f'Unknown model type: {model_type}')

def total_score(bert_scores, bert_weight, uni_sent_encoder_scores, uni_sent_encoder_weight):
    '''
    Create a total score which will be used for MAP evaluation.
    '''
    final_score = {}

    for doc_id in bert_scores:
        final_score[doc_id] = bert_scores[doc_id] * bert_weight
    
    for doc_id in uni_sent_encoder_scores:
        if doc_id in final_score:
            final_score[doc_id] += uni_sent_encoder_scores[doc_id] * uni_sent_encoder_weight
        else:
            final_score[doc_id] = uni_sent_encoder_scores[doc_id] * uni_sent_encoder_weight

    return final_score

def neural_rank_documents(model_type, model_name, documents, inverted_index, documents_length, queries, reranking):
    '''
    Rank the documents using the specified model.
    '''
    model = create_model(model_type, model_name, documents, inverted_index, documents_length)

    # Build a corpus dictionary from the documents
    corpus = {}
    for doc in documents:
        corpus[doc['DOCNO']] = {
            'title': ' '.join(doc['HEAD']),
            'text': ' '.join(doc['TEXT'])
        }

    # Convert queries into the expected dictionary format.
    # For BM25, pass the tokens as a list; for dense models, join them into a string.
    query_dict = {}
    for query in queries:
        tokens = query.get('title', []) + query.get('query', []) + query.get('narrative', [])
        query_dict[query['num']] = ' '.join(tokens)
    
    # retrieve results
    print("Ranking top documents for all queries and creating associated file")
    writeResults("../Results_Scores/BM25/TopScoresAllQueries.txt", queries, model, "top_scores_run")

    print("\nRanking top 10 documents for the first 2 queries and creating associated file")
    writeResultsTop10First2("../Results_Scores/BM25/Top10AnswersFirst2Queries.txt", queries, model, "top_10_first_2_run")

    print("\nRanking all documents for all queries and creating associated file")
    writeResultsAll("../Results_Scores/BM25/AllScoresAllQueries.txt", queries, model, "all_scores_run")

    print("\nRanking top 100 documents for all queries and creating associated file. This is the file that will be used for final evaluation.")
    results = writeResultsTop100("../Results_Scores/BM25/Results.txt", queries, model, "top_100_best_run")
        
    # Refine results with re-ranking using a CROSSENCODER with different models
    if reranking == "BERT":
        BERT_model = create_model("cross-encoder", "cross-encoder/nli-distilroberta-base", None, None, None)
        reranker = CustomRerank(BERT_model, batch_size=128)
        results = reranker.rerank(corpus, query_dict, results, top_k=100)
    
    elif reranking == "ELECTRA":
        ELECTRA_model = create_model("cross-encoder", "cross-encoder/ms-marco-electra-base", None, None, None)
        reranker = Rerank(ELECTRA_model, batch_size=128)
        results = reranker.rerank(corpus, query_dict, results, top_k=100)
    
    return results

def neural_save_results(results, output_file):
    lines = []
    for query_id, docs in results.items():
        # Check if docs is a list or dict
        if isinstance(docs, list):
            # If it's a list, sort the list of (doc_id, score) tuples.
            ranked_list = sorted(docs, key=lambda x: x[1], reverse=True)
        else:
            # Otherwise, if it's a dict, sort its items.
            ranked_list = sorted(docs.items(), key=lambda x: x[1], reverse=True)
        
        normalized_ranked = normalize_neural(ranked_list)
        
        for rank, (doc_id, score) in enumerate(normalized_ranked, start=1):
            line = f"{query_id} Q0 {doc_id} {rank} {score} top_100_best_run"
            lines.append(line)
    
    # Write all lines to the output file.
    with open(output_file, 'w') as file:
        file.write("\n".join(lines))

def normalize_neural(ranked_docs):
    """
    Normalize scores to [0,1] range.
    """
    if not ranked_docs:
        return []
    max_score = max(score for _, score in ranked_docs)
    min_score = min(score for _, score in ranked_docs)
    if max_score == min_score:
        return [(doc_id, 1.0) for doc_id, _ in ranked_docs]
    return [(doc_id, (score - min_score) / (max_score - min_score)) for doc_id, score in ranked_docs]
