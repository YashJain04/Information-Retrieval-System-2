# imports
import tensorflow as tf  # Import tf first
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
model_type = "UNI_SENT_ENCODER"
model_name = "universal-sentence-encoder-qa/1"
# model_type = "BM25"
# model_name = None
neural_results = neural_rank_documents(model_type, model_name, documents, inverted_index, doc_lengths, queries, True)
end_time = time.time() # end the timer
results_file = '../Results_Scores/NeuralUSCScores.txt' # get results file
neural_save_results(neural_results, results_file) # save the results
print(f"Ranking results written to {results_file}") # print results
print(f"\nTime taken to complete STEP 4 (NEURAL RANKING): {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# # STEP 5 - Computing MAP Scores Through PYTREC_EVAL
# print("---------------------------------------------------------------------------------------")
# print("\nRunning The TREC_EVAL to retrieve the MAP Scores")

# def read_qrel(file_path):
#     '''
#     Read the test file (test.tsv) and store it in qrel
#     '''
#     qrel = {}
#     with open(file_path, 'r') as f:
#         # skip the header
#         header = f.readline()  # read and discard the header
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) == 3:
#                 query_id, doc_id, relevance = parts
#                 relevance = int(relevance)  # convert relevance to an integer
#                 if query_id not in qrel:
#                     qrel[query_id] = {}
#                 qrel[query_id][doc_id] = relevance
#     return qrel

# def read_run(file_path):
#     with open(file_path, 'r') as f:
#         # Skip the first 325219 lines
#         for _ in range(325219):
#             next(f)
#         # Read the remaining content
#         remaining = f.read().strip()
        
#     # If the remaining content does not start with '{', add curly braces to form valid JSON
#     if not remaining.startswith('{'):
#         json_str = "{" + remaining
#         if not remaining.endswith('}'):
#             json_str += "}"
#     else:
#         json_str = remaining

#     # Load the JSON content
#     run = json.loads(json_str)
    
#     new_run = {}
#     for query_id, docs in run.items():
#         # Sort the list of [doc_id, score] pairs in descending order of score, then take the top 100
#         sorted_docs = sorted(docs, key=lambda x: x[1], reverse=True)[:100]
#         # Convert sorted list into a dictionary {doc_id: score}
#         new_run[query_id] = {doc_id: score for doc_id, score in sorted_docs}
        
#     return new_run


# # get the file paths
# qrel_file = "../scifact/qrels/test.tsv"
# run_file = '../Results_Scores/NeuralScores.txt'

# # read the files (qrel) and (run)
# qrel = read_qrel(qrel_file)
# run = read_run(run_file)

# # evaluate using pytrec_eval
# evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg'})
# results = evaluator.evaluate(run)

# # save the results to a file
# output_file = '../Results_Scores/MAPBertScores.json'
# with open(output_file, 'w') as f:
#     json.dump(results, f, indent=1)

# # return the path to the results file
# print(f"Evaluation results saved to {output_file}")

# # get the average MAP score
# total_map = sum(results[query]['map'] for query in results) / len(results)  # average map scores for all queries

# # round to 3 decimal places
# total_map = round(total_map, 8)

# # print the average map score
# print("The average MAP Score is: ", total_map)
# print("STEP 5 COMPLETE")
# print("")
# print("---------------------------------------------------------------------------------------")