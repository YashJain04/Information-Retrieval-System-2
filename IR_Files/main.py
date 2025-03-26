import tensorflow as tf
from parser import *
from preprocessing import *
from indexing import *
from neural_ranking import *
import time
import pytrec_eval
import json

# file directories
doc_folder_path = '../scifact/corpus.jsonl'
query_file_path = '../scifact/queries.jsonl'
index_file_path = '../Results_Scores/Building/inverted_index.json'
preprocessed_docs_path = '../Results_Scores/Building/preprocessed_documents.json'
preprocessed_queries_path = '../Results_Scores/Building/preprocessed_queries.json'

# STEP 0 - Parse the document
start_time = time.time() # start the timer
print("---------------------------------------------------------------------------------------")
print("")
print("Parsing documents")
documents = []
queries = parse_queries_from_file(query_file_path)
end_time = time.time() # end the timer
print(f"Time taken to complete STEP 0 (PARSING DOCS): {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# STEP 1 - Preprocess documents and queries
start_time = time.time() # start the timer
print("---------------------------------------------------------------------------------------")
print("")
print("Preprocessing documents")
documents = parse_documents_from_file(doc_folder_path)
documents = preprocess_documents(documents)
# documents = preprocess_documents_head_only(documents)
# if you want to preprocess only the head of the documents, use the above line instead, and uncomment the line above it
documents = preprocess_documents(parse_documents_from_file(doc_folder_path))
save_preprocessed_data(documents, preprocessed_docs_path)
print("Preprocessing queries")
queries = preprocess_queries(parse_queries_from_file(query_file_path))
save_preprocessed_data(queries, preprocessed_queries_path)
print("The length of the vocabulary is", len(documents))
with open("../Results_Scores/Building/Sample100Vocabulary.txt", "w", encoding="utf-8") as f: # save 100 Vocabulary
    json.dump(documents[:100], f, indent=4)
end_time = time.time() # end the timer
print(f"Time taken to complete STEP 1 (PREPROCESS DOCS/QUERIES): {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# STEP 2 - Build or load inverted index
start_time = time.time() # start the timer
print("---------------------------------------------------------------------------------------")
print("")
print("Building an inverted index.")
inverted_index = build_inverted_index(documents)
# inverted_index = build_inverted_index_head_only(documents)
# if you want to build the inverted index using only the head of the documents, use the above line instead, and uncomment the line above it
save_inverted_index(inverted_index, index_file_path)
end_time = time.time() # end the timer
print(f"Time taken to complete STEP 2 (BUILD INVERTED INDEX): {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# STEP 3 - Getting Document Lengths
start_time = time.time() # start the timer
print("---------------------------------------------------------------------------------------")
print("")
print("Getting document lengths.")
doc_lengths = calculate_document_lengths(documents)
# doc_lengths = calculate_document_lengths_head_only(documents)
# if you want to calculate the document lengths using only the head of the documents, use the above line instead, and uncomment the line above it
end_time = time.time() # end the timer
print(f"Time taken to complete STEP 3 (GETTING DOC_LENGTHS): {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# STEP 4.0 - Initial Ranking (A1 IR SYSTEM - BM25)
start_time = time.time() # start the timer
print("---------------------------------------------------------------------------------------")
print("")
print("Computing BM25 Ranking. This is our Initial IR System from Assignment 1.")
model_type = "BM25"
model_name = None
neural_rank_documents(model_type, model_name, documents, inverted_index, doc_lengths, queries, None)
end_time = time.time() # end the timer
print(f"\nTime taken to complete STEP 4.0 - BM25 Model Ranking: {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# STEP 4.1 - Neural Ranking (BERT RERANK)
start_time = time.time() # start the timer
print("---------------------------------------------------------------------------------------")
print("")
print("Computing BM25 Ranking with RE-RANKING using a BERT MODEL.")
model_type = "BM25"
model_name = None
neural_results = neural_rank_documents(model_type, model_name, documents, inverted_index, doc_lengths, queries, "BERT")
neural_save_results(neural_results, "../Results_Scores/BERT/Results.txt")
end_time = time.time() # end the timer
print(f"\nTime taken to complete STEP 4.1 - BERT MODEL RE-RANKING: {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# STEP 4.2 - Neural Ranking (ELECTRA RERANK)
start_time = time.time() # start the timer
print("---------------------------------------------------------------------------------------")
print("")
print("Computing BM25 Ranking with RE-RANKING using an ELECTRA MODEL.")
model_type = "BM25"
model_name = None
neural_results = neural_rank_documents(model_type, model_name, documents, inverted_index, doc_lengths, queries, "ELECTRA")
neural_save_results(neural_results, "../Results_Scores/ELECTRA/Results.txt")
end_time = time.time() # end the timer
print(f"\nTime taken to complete STEP 4.2 - ELECTRA MODEL RE-RANKING: {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# STEP 5 - Computing MAP Scores Through PYTREC_EVAL
print("---------------------------------------------------------------------------------------")
print("\nRunning The TREC_EVAL to retrieve the MAP Scores")

def read_qrel(file_path):
    '''
    Read the test file (test.tsv) and store it in qrel
    '''
    qrel = {}
    with open(file_path, 'r') as f:
        # skip the header
        header = f.readline()  # read and discard the header
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                query_id, doc_id, relevance = parts
                relevance = int(relevance)  # convert relevance to an integer
                if query_id not in qrel:
                    qrel[query_id] = {}
                qrel[query_id][doc_id] = relevance
    return qrel

def read_run(file_path):
    '''
    Read the results file (Results.txt) and store it in run,
    skipping the first 80,735 lines to get the test queries.
    '''
    run = {}
    with open(file_path, 'r') as f:
        # skip the first 80,735 lines and start at the line where the first query is 1 (these are the test.tsv queries)
        for _ in range(80735):
            next(f)
        
        # process the rest of the file
        for line in f:
            parts = line.strip().split()  # split by any whitespace
            if len(parts) >= 6:  # ensure we have at least 6 columns
                query_id, _, doc_id, rank, score, _ = parts[:6]
                score = float(score)
                if query_id not in run:
                    run[query_id] = {}
                run[query_id][doc_id] = score

    return run

# get the file paths
qrel_file = "../scifact/qrels/test.tsv"
run_bm25 = "../Results_Scores/BM25/Results.txt"
run_bert = "../Results_Scores/BERT/Results.txt"
run_electra = "../Results_Scores/ELECTRA/Results.txt"

# read the files (qrel) and (run)
qrel = read_qrel(qrel_file)
run_bm25 = read_run(run_bm25)
run_bert = read_run(run_bert)
run_electra = read_run(run_electra)

# evaluate using pytrec_eval
evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg'})
results_bm25 = evaluator.evaluate(run_bm25)
results_bert = evaluator.evaluate(run_bert)
results_electra = evaluator.evaluate(run_electra)

# save the results to a file
output_file_bm25 = '../Results_Scores/MAP/BM25_MAP_SCORE.json'
output_file_bert = '../Results_Scores/MAP/BERT_MAP_SCORE.json'
output_file_electra = '../Results_Scores/MAP/ELECTRA_MAP_SCORE.json'

with open(output_file_bm25, 'w') as f:
    json.dump(results_bm25, f, indent=1)

with open(output_file_bert, 'w') as f:
    json.dump(results_bert, f, indent=1)

with open(output_file_electra, 'w') as f:
    json.dump(results_electra, f, indent=1)

# return the path to the results file
print(f"Evaluation results for BM25 model saved to {output_file_bm25}")
print(f"Evaluation results for BERT model saved to {output_file_bert}")
print(f"Evaluation results for ELECTRA model saved to {output_file_electra}")

# get the average MAP score - average map scores for all queries
total_map_bm25 = sum(results_bm25[query]['map'] for query in results_bm25) / len(results_bm25)
total_map_bert = sum(results_bert[query]['map'] for query in results_bert) / len(results_bert)
total_map_electra = sum(results_electra[query]['map'] for query in results_electra) / len(results_electra)

# round to 4 decimal places
total_map_bm25 = round(total_map_bm25, 4)
total_map_bert = round(total_map_bert, 4)
total_map_electra = round(total_map_electra, 4)

# print the average map score
print("The average MAP Score for the INITIAL BM25 MODEL is : ", total_map_bm25)
print("The average MAP Score for the BERT MODEL is: ", total_map_bert)
print("The average MAP Score for the ELECTRA MODEL is: ", total_map_electra)
print("STEP 5 COMPLETE")
print("")
print("---------------------------------------------------------------------------------------")