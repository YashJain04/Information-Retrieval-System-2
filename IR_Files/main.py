# imports
import time
from parser import *
from preprocessing import *
from indexing import *
from ranking import *
from utils import *
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
# documents = preprocess_documents_head_only(documents)
# if you want to preprocess only the head of the documents, use the above line instead, and uncomment the line above it

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
# inverted_index = build_inverted_index_head_only(documents)
# if you want to build the inverted index using only the head of the documents, use the above line instead, and uncomment the line above it

save_inverted_index(inverted_index, index_file_path)
end_time = time.time() # end the timer
print(f"Time taken to complete STEP 2 (BUILD INVERTED INDEX): {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")


# STEP 3 - Retrieval and ranking
print("---------------------------------------------------------------------------------------")
print("")
start_time = time.time() # start the timer

doc_lengths = calculate_document_lengths(documents)
# doc_lengths = calculate_document_lengths_head_only(documents)
# if you want to calculate the document lengths using only the head of the documents, use the above line instead, and uncomment the line above it

beir_results = {}
print("Ranking and writing to results file")
bm25 = BM25(inverted_index, doc_lengths)

print("Ranking top documents for all queries and creating associated file")
writeResults("../Results_Scores/TopScoresAllQueries.txt", queries, bm25, "top_scores_run")

print("\nRanking top 10 documents for the first 2 queries and creating associated file")
writeResultsTop10First2("../Results_Scores/Top10AnswersFirst2Queries.txt", queries, bm25, "top_10_first_2_run")

print("\nRanking all documents for all queries and creating associated file")
writeResultsAll("../Results_Scores/AllScoresAllQueries.txt", queries, bm25, "all_scores_run")

print("\nRanking top 100 documents for all queries and creating associated file. This is the file that will be used for final evaluation.")
writeResultsTop100("../Results_Scores/Results.txt", queries, bm25, "top_100_best_run")

end_time = time.time() # end the timer
print(f"\nTime taken to complete STEP 3 (RANKING DOCUMENTS): {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# STEP 4 - Return results
print("---------------------------------------------------------------------------------------")
print(f"\nRANKING: Top Documents For Each Query Written To:", "../Results_Scores/TopScoresAllQueries.txt")
print(f"RANKING: Top 10 Documents For First 2 Queries Written To:", "../Results_Scores/Top10AnswersFirst2Queries.txt")
print(f"RANKING: All Documents For Each Query Written To:", "../Results_Scores/AllScoresAllQueries.txt")
print("STEP 4 COMPLETE")
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
run_file = '../Results_Scores/Results.txt'

# read the files (qrel) and (run)
qrel = read_qrel(qrel_file)
run = read_run(run_file)

# evaluate using pytrec_eval
evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg'})
results = evaluator.evaluate(run)

# save the results to a file
output_file = '../Results_Scores/MAPScores.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=1)

# return the path to the results file
print(f"Evaluation results saved to {output_file}")

# get the average MAP score
total_map = sum(results[query]['map'] for query in results) / len(results)  # average map scores for all queries

# round to 3 decimal places
total_map = round(total_map, 8)

# print the average map score
print("The average MAP Score is: ", total_map)
print("STEP 5 COMPLETE")
print("")
print("---------------------------------------------------------------------------------------")