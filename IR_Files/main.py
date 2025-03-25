# imports
import time
from parser import *
from preprocessing import *
from indexing import *
from ranking import *
from utils import *
from neural_ranking import *
import pytrec_eval
import json

# file directories
doc_folder_path = '../scifact/corpus.jsonl'
query_file_path = '../scifact/queries.jsonl'
index_file_path = '../Results_Scores/inverted_index.json'
preprocessed_docs_path = '../Results_Scores/preprocessed_documents.json'
preprocessed_queries_path = '../Results_Scores/preprocessed_queries.json'

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
print("---------------------------------------------------------------------------------------")
print("")
start_time = time.time() # start the timer
print("Preprocessing documents")
documents = parse_documents_from_file(doc_folder_path)
documents = preprocess_documents(documents)
documents = preprocess_documents(parse_documents_from_file(doc_folder_path))
save_preprocessed_data(documents, preprocessed_docs_path)
print("Preprocessing queries")
queries = preprocess_queries(parse_queries_from_file(query_file_path))
save_preprocessed_data(queries, preprocessed_queries_path)
print("The length of the vocabulary is", len(documents))

with open("../Results_Scores/Sample100Vocabulary.txt", "w", encoding="utf-8") as f:
    json.dump(documents[:100], f, indent=4)  # Pretty-print for readability

end_time = time.time() # end the timer
print(f"Time taken to complete STEP 1 (PREPROCESS DOCS/QUERIES): {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# STEP 2 - Build or load inverted index
print("---------------------------------------------------------------------------------------")
print("")
start_time = time.time() # start the timer
print("Building an inverted index.")
inverted_index = build_inverted_index(documents)
save_inverted_index(inverted_index, index_file_path)
end_time = time.time() # end the timer
print(f"Time taken to complete STEP 2 (BUILD INVERTED INDEX): {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# STEP 3 - Getting Document Lengths
print("---------------------------------------------------------------------------------------")
print("")
start_time = time.time() # start the timer
doc_lengths = calculate_document_lengths(documents)
end_time = time.time() # end the timer
print(f"\nTime taken to complete STEP 3 (GETTING DOC_LENGTHS): {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# STEP 4 - Neural Ranking
print("---------------------------------------------------------------------------------------")
print("")
start_time = time.time() # start the timer
model_type = "BERT"
model_name = "msmarco-distilbert-base-v3"
# model_type = "BM25"
# model_name = None
neural_results = neural_rank_documents(model_type, model_name, documents, inverted_index, doc_lengths, queries, False)
end_time = time.time() # end the timer
results_file = '../Results_Scores/NeuralScores.txt' # get results file
neural_save_results(neural_results, results_file) # save the results
print(f"Ranking results written to {results_file}") # print results
print(f"\nTime taken to complete STEP 4 (NEURAL RANKING): {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")