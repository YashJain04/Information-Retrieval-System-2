import json
from ranking import BM25
from ranking import normalize_scores
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank

def create_model(model_type, model_name, documents, inverted_index, documents_length):
    '''
    Load a specific model in [BM25, BERT, UNIVERSAL SENTENCE ENCODER]
    '''
    # INITIAL our model
    if model_type == 'BM25':
        return BM25(inverted_index, documents_length)
    
    # BERT model
    elif model_type == 'BERT':
        return DRES(models.SentenceBERT(model_name), batch_size=16)
    
    # UNIVERSAL SENTENCE ENCODER model
    elif model_type == 'UNI_SENT_ENCODER':
        return DRES(models.UseQA(model_name))
    
    # CROSS ENCODER model
    elif model_type == 'cross_encoder':
        return CrossEncoder(model_name)
    
    # INVALID model
    else:
        print('This is an unknown model type. Ending program.')
        raise ValueError(f'Unknown model type: {model_type}')

def total_score(bert_scores, bert_weight, uni_sent_encoder_scores, uni_sent_encoder_weight):
    '''
    Create a total score which will be used for MAP evaluation
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
    Rank the documents
    '''
    model = create_model(model_type, model_name, documents, inverted_index, documents_length)
    
    # Build a corpus dictionary from the documents
    corpus = {}
    for doc in documents:
        corpus[doc['DOCNO']] = {
            'title': ' '.join(doc['HEAD']),
            'text': ' '.join(doc['TEXT'])
        }
        
    # Determine which scoring function to use
    scoring = 'cos_sim'
    
    # Initialize the retriever with the chosen score function
    if model_type != 'BM25':
        retriever = EvaluateRetrieval(model, score_function = scoring)
    else:
        retriever = EvaluateRetrieval(model)
    
    # Convert queries into the expected dictionary format
    query_dict = {}
    for query in queries:
        title = query.get('title', [])
        query_text = query.get('query', [])
        narrative = query.get('narrative', [])
        query_dict[query['num']] = ' '.join(title + query_text + narrative)
    
    # Retrieve results based on model type
    if model_type != 'BM25':
        results = retriever.retrieve(corpus, query_dict)
    else:
        results = model.neural(corpus, query_dict)
        
    # Refine results with re-ranking using a CROSSENCODER
    if reranking:
        cross_encoder_model = create_model("cross-encoder", "cross-encoder/ms-marco-electra-base", None, None, None)
        reranker = Rerank(cross_encoder_model, batch_size=128)
        results = reranker.rerank(corpus, query_dict, results, top_k=100)
    
    return results

def neural_save_results(results, output_file):
    final_results = {}
    for query_id, docs in results.items():
        final_results[query_id] = [(doc_id, float(score)) for doc_id, score in docs.items()]
    
    with open(output_file, 'w') as file:
        json.dump(final_results, file, indent=4)