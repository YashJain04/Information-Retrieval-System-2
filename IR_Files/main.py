# imports
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
neural_rank_documents(model_type, model_name, documents, inverted_index, doc_lengths, queries, False)
end_time = time.time() # end the timer
print(f"\nTime taken to complete STEP 4.0 - BM25 Model Ranking: {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# STEP 4.1 - Neural Ranking (BERT)
start_time = time.time() # start the timer
print("---------------------------------------------------------------------------------------")
print("")
print("Computing BERT Ranking. This is our BERT MODEL.")
model_type = "BERT"
model_name = "msmarco-distilbert-base-v3"
neural_results = neural_rank_documents(model_type, model_name, documents, inverted_index, doc_lengths, queries, False)
neural_save_results(neural_results, "../Results_Scores/BERT/Results.txt")
end_time = time.time() # end the timer
print(f"\nTime taken to complete STEP 4.1 - BERT Model Ranking: {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")