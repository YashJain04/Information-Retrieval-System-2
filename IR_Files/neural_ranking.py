# imports
from ranking import BM25
from customizer import CustomRerank
from utils import *
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank
from sentence_transformers import SentenceTransformer, util

def create_model(model_type, model_name, documents, inverted_index, documents_length):
    """
    load a specific model in [BM25, ELECTRA, MINILM, BERT]
    """
    if model_type == 'BM25':
        return BM25(inverted_index, documents_length)
    
    elif model_type == 'cross-encoder':
        return CrossEncoder(model_name)
    
    elif model_type == 'mini-lm':
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    else:
        raise ValueError(f'Unknown model type: {model_type}')

def total_score(score_dict1, weight1, score_dict2, weight2):
    """
    create a total score which will be used for MAP evaluation by combining two sets of scores that have weights
    """
    try:
        w1 = float(weight1)
        w2 = float(weight2)
    except Exception as e:
        raise ValueError("Weights must be numbers.") from e

    final_score = {}
    for doc_id, score in score_dict1.items():
        final_score[doc_id] = score * w1
    for doc_id, score in score_dict2.items():
        if doc_id in final_score:
            final_score[doc_id] += score * w2
        else:
            final_score[doc_id] = score * w2
    return final_score

def neural_rank_documents(model_type, model_name, documents, inverted_index, documents_length, queries, reranking):
    """
    rank the documents using the specified model for both title and text
    """

    # initial retrieval using BM25
    model = create_model(model_type, model_name, documents, inverted_index, documents_length)

    # build a corpus dictionary from the documents using full TITLE+TEXT
    corpus = {}
    for doc in documents:
        corpus[doc['DOCNO']] = {
            'title': ' '.join(doc['HEAD']),
            'text': ' '.join(doc['TEXT'])
        }

    # convert queries into the expected dictionary format
    query_dict = {}
    for query in queries:
        tokens = query.get('title', []) + query.get('query', []) + query.get('narrative', [])
        query_dict[query['num']] = ' '.join(tokens)
    
    # retrieve BM25 results (the same regardless of re-ranking type)
    print("Ranking documents and showing progress bars. If for some reason progress bars get stuck, note that it is just a visual glitch. Everything is indeed loaded correctly.")
    
    print("Ranking top documents for all queries and creating associated file")
    writeResults("../Results_Scores/BM25/TopScoresAllQueries.txt", queries, model, "top_scores_run")

    print("\nRanking top 10 documents for the first 2 queries and creating associated file")
    writeResultsTop10First2("../Results_Scores/BM25/Top10AnswersFirst2Queries.txt", queries, model, "top_10_first_2_run")

    print("\nRanking all documents for all queries and creating associated file")
    writeResultsAll("../Results_Scores/BM25/AllScoresAllQueries.txt", queries, model, "all_scores_run")

    print("\nRanking top 100 documents for all queries and creating associated file. This is the file that will be used for final evaluation.")
    bm25_results = writeResultsTop100("../Results_Scores/BM25/Results.txt", queries, model, "top_100_best_run")
        
    # refine results with re-ranking using a dense model from sentence-transformers

    # MINI LM model
    if reranking == "MINI_LM":
        print("\nUsing MINI LM model.")
        mini_lm_model = create_model("mini-lm", "sentence-transformers/all-MiniLM-L6-v2", None, None, None)
        bm25_weight = 0.4
        mini_lm_weight = 0.6
        dense_scores = {}
        print("Iterating over the results retrieved from the initial IR system.")
        for query_id, candidate_dict in bm25_results.items():
            query_text = query_dict[query_id]
            query_embedding = mini_lm_model.encode(query_text, convert_to_tensor=True)
            candidate_doc_ids = []
            candidate_texts = []
            for doc_id, _ in candidate_dict.items():
                candidate_doc_ids.append(doc_id)
                candidate_texts.append(corpus[doc_id]['title'] + " " + corpus[doc_id]['text']) # use both title and text for richer representation
            doc_embeddings = mini_lm_model.encode(candidate_texts, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0].tolist()
            dense_scores[query_id] = {doc_id: score for doc_id, score in zip(candidate_doc_ids, cos_scores)}
        
        print("Using weighted sums now.")
        combined_results = {}
        for query_id, candidate_dict in bm25_results.items():
            bm25_dict = candidate_dict
            combined = total_score(bm25_dict, bm25_weight, dense_scores[query_id], mini_lm_weight)
            combined_results[query_id] = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        print("Results are being returned.")
        results = combined_results
    
    # ELECTRA model
    elif reranking == "ELECTRA":
        print("\nUsing ELECTRA model.")
        ELECTRA_model = create_model("cross-encoder", "cross-encoder/ms-marco-electra-base", None, None, None)
        reranker = Rerank(ELECTRA_model, batch_size=128)
        results = reranker.rerank(corpus, query_dict, bm25_results, top_k=100)

    # BERT model
    elif reranking == "BERT":
        print("\nUsing BERT model.")
        BERT_model = create_model("cross-encoder", "cross-encoder/nli-distilroberta-base", None, None, None)
        reranker = CustomRerank(BERT_model, batch_size=128)
        results = reranker.rerank(corpus, query_dict, bm25_results, top_k=100)
    
    # no neural re-ranking is set just use the base results from BM25
    else:
        results = bm25_results
    
    return results

def neural_rank_documents_head_only(model_type, model_name, documents, inverted_index, documents_length, queries, reranking):
    """
    rank the documents using the specified model for only title
    """

    # initial retrieval using BM25
    model = create_model(model_type, model_name, documents, inverted_index, documents_length)

    # build a corpus dictionary from the documents using only TITLE
    corpus = {}
    for doc in documents:
        corpus[doc['DOCNO']] = {
            'title': ' '.join(doc['HEAD'])
        }

    # convert queries into the expected dictionary format
    query_dict = {}
    for query in queries:
        tokens = query.get('title', []) + query.get('query', []) + query.get('narrative', [])
        query_dict[query['num']] = ' '.join(tokens)
    
    # retrieve BM25 results (the same regardless of re-ranking type)
    print("Ranking documents and showing progress bars. If for some reason progress bars get stuck, note that it is just a visual glitch. Everything is indeed loaded correctly.")

    print("Ranking top documents for all queries and creating associated file")
    writeResults("../Results_Scores/BM25/TopScoresAllQueries_head_only.txt", queries, model, "top_scores_run")
    
    print("\nRanking top 10 documents for the first 2 queries and creating associated file")
    writeResultsTop10First2("../Results_Scores/BM25/Top10AnswersFirst2Queries_head_only.txt", queries, model, "top_10_first_2_run")
    
    print("\nRanking all documents for all queries and creating associated file")
    writeResultsAll("../Results_Scores/BM25/AllScoresAllQueries_head_only.txt", queries, model, "all_scores_run")
    
    print("\nRanking top 100 documents for all queries and creating associated file.")
    bm25_results = writeResultsTop100("../Results_Scores/BM25/Results_head_only.txt", queries, model, "top_100_best_run")
    
    # refine results with re-ranking using a dense model from sentence-transformers

    # MINI LM model
    if reranking == "MINI_LM":
        print("\nUsing MINI LM model. This is for only TITLE/HEAD.")
        mini_lm_model = create_model("mini-lm", "sentence-transformers/all-MiniLM-L6-v2", None, None, None)
        bm25_weight = 0.4
        mini_lm_weight = 0.6
        dense_scores = {}
        for query_id, candidate_dict in bm25_results.items():
            query_text = query_dict[query_id]
            query_embedding = mini_lm_model.encode(query_text, convert_to_tensor=True)
            candidate_doc_ids = []
            candidate_texts = []
            for doc_id, _ in candidate_dict.items():
                candidate_doc_ids.append(doc_id)
                candidate_texts.append(corpus[doc_id]['title']) # use only the title for representation
            doc_embeddings = mini_lm_model.encode(candidate_texts, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0].tolist()
            dense_scores[query_id] = {doc_id: score for doc_id, score in zip(candidate_doc_ids, cos_scores)}
        
        combined_results = {}
        for query_id, candidate_dict in bm25_results.items():
            bm25_dict = candidate_dict
            combined = total_score(bm25_dict, bm25_weight, dense_scores[query_id], mini_lm_weight)
            combined_results[query_id] = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        results = combined_results
    
    # ELECTRA model
    elif reranking == "ELECTRA":
        print("\nUsing ELECTRA model. This is for only TITLE/HEAD.")
        ELECTRA_model = create_model("cross-encoder", "cross-encoder/ms-marco-electra-base", None, None, None)
        reranker = CustomRerank(ELECTRA_model, batch_size=128) # the standard re-ranker is NOT used because it’s designed to work with richer document representations (i.e. both title and text)
        results = reranker.rerank_head_only(corpus, query_dict, bm25_results, top_k=100)

    # BERT model
    elif reranking == "BERT":
        print("\nUsing BERT model. This is for only TITLE/HEAD.")
        BERT_model = create_model("cross-encoder", "cross-encoder/nli-distilroberta-base", None, None, None)
        reranker = CustomRerank(BERT_model, batch_size=128)
        results = reranker.rerank_head_only(corpus, query_dict, bm25_results, top_k=100)
    
    else:
        results = bm25_results
    
    return results

def neural_save_results(results, output_file):
    '''
    save the results to a file path from the neural ranking models
    '''
    lines = []
    for query_id, docs in results.items():
        if isinstance(docs, list):
            ranked_list = sorted(docs, key=lambda x: x[1], reverse=True)
        else:
            ranked_list = sorted(docs.items(), key=lambda x: x[1], reverse=True)
        normalized_ranked = normalize_neural(ranked_list)
        for rank, (doc_id, score) in enumerate(normalized_ranked, start=1):
            line = f"{query_id} Q0 {doc_id} {rank} {score} top_100_best_run"
            lines.append(line)
    with open(output_file, 'w') as file:
        file.write("\n".join(lines))

def normalize_neural(ranked_docs):
    """
    normalize the scores in range [0, 1]
    """
    if not ranked_docs:
        return []
    max_score = max(score for _, score in ranked_docs)
    min_score = min(score for _, score in ranked_docs)
    if max_score == min_score:
        return [(doc_id, 1.0) for doc_id, _ in ranked_docs]
    return [(doc_id, (score - min_score) / (max_score - min_score)) for doc_id, score in ranked_docs]