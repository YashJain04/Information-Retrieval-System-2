# imports
from parser import *
from preprocessing import *
from indexing import *
from neural_ranking import *
import time
import pytrec_eval
import json
import os

# file directories
doc_folder_path = '../scifact/corpus.jsonl'
query_file_path = '../scifact/queries.jsonl'
index_file_path = '../Results_Scores/Building/inverted_index.json'
preprocessed_docs_path = '../Results_Scores/Building/preprocessed_documents.json'
preprocessed_queries_path = '../Results_Scores/Building/preprocessed_queries.json'

# ===============================================================================================================================================

# STEP 0 - document parsing
start_time = time.time()
print("---------------------------------------------------------------------------------------")
print("")
print("Parsing documents")
documents = []
queries = parse_queries_from_file(query_file_path)
end_time = time.time()
print(f"Time taken to complete STEP 0 (PARSING DOCS): {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# ===============================================================================================================================================

# STEP 1 - preprocessing documents and queries
start_time = time.time()
print("---------------------------------------------------------------------------------------")
print("")
print("Preprocessing documents")
documents = parse_documents_from_file(doc_folder_path)
documents = preprocess_documents(documents)
documents = preprocess_documents(parse_documents_from_file(doc_folder_path))
save_preprocessed_data(documents, preprocessed_docs_path)
print("Preprocessing queries")
queries = preprocess_queries(parse_queries_from_file(query_file_path))
save_preprocessed_data(queries, preprocessed_queries_path)
print("The length of the vocabulary is", len(documents))
with open("../Results_Scores/Building/Sample100Vocabulary.txt", "w", encoding="utf-8") as f: # save 100 vocabulary
    json.dump(documents[:100], f, indent=4)
end_time = time.time()
print(f"Time taken to complete STEP 1 (PREPROCESS DOCS/QUERIES): {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# ===============================================================================================================================================

# STEP 2 - building the inverted index
start_time = time.time()
print("---------------------------------------------------------------------------------------")
print("")
print("Building an inverted index.")
inverted_index = build_inverted_index(documents)
save_inverted_index(inverted_index, index_file_path)
end_time = time.time()
print(f"Time taken to complete STEP 2 (BUILD INVERTED INDEX): {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# ===============================================================================================================================================

# STEP 3 - calculate document lengths
start_time = time.time()
print("---------------------------------------------------------------------------------------")
print("")
print("Getting document lengths.")
doc_lengths = calculate_document_lengths(documents)
end_time = time.time()
print(f"Time taken to complete STEP 3 (GETTING DOC_LENGTHS): {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# ===============================================================================================================================================

# STEP 4.0 - initial ranking with BM25 model
start_time = time.time()
print("---------------------------------------------------------------------------------------")
print("")
print("Computing BM25 Ranking. This is our Initial IR System from Assignment 1.")
model_type = "BM25"
model_name = None
neural_rank_documents(model_type, model_name, documents, inverted_index, doc_lengths, queries, None)
end_time = time.time()
print(f"\nTime taken to complete STEP 4.0 - BM25 Model Ranking: {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# ===============================================================================================================================================

# STEP 4.1 - neural ranking with ELECTRA model
start_time = time.time()
print("---------------------------------------------------------------------------------------")
print("")
print("Computing BM25 Ranking with RE-RANKING using an ELECTRA MODEL.")
model_type = "BM25"
model_name = None
neural_results = neural_rank_documents(model_type, model_name, documents, inverted_index, doc_lengths, queries, "ELECTRA")
neural_save_results(neural_results, "../Results_Scores/ELECTRA/Results.txt")
end_time = time.time()
print(f"\nTime taken to complete STEP 4.1 - ELECTRA MODEL RE-RANKING: {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# ===============================================================================================================================================

# STEP 4.2 - neural ranking with MINI LM model
start_time = time.time()
print("---------------------------------------------------------------------------------------")
print("")
print("Computing BM25 Ranking with RE-RANKING using a MINI LM MODEL.")
model_type = "BM25"
model_name = None
neural_results = neural_rank_documents(model_type, model_name, documents, inverted_index, doc_lengths, queries, "MINI_LM")
neural_save_results(neural_results, "../Results_Scores/MINI_LM/Results.txt")
end_time = time.time()
print(f"\nTime taken to complete STEP 4.2 - MINI_LM MODEL RE-RANKING: {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# ===============================================================================================================================================

# STEP 4.3 - neural ranking with BERT model
start_time = time.time()
print("---------------------------------------------------------------------------------------")
print("")
print("Computing BM25 Ranking with RE-RANKING using a BERT MODEL.")
model_type = "BM25"
model_name = None
neural_results = neural_rank_documents(model_type, model_name, documents, inverted_index, doc_lengths, queries, "BERT")
neural_save_results(neural_results, "../Results_Scores/BERT/Results.txt")
end_time = time.time()
print(f"\nTime taken to complete STEP 4.3 - BERT MODEL RE-RANKING: {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# ===============================================================================================================================================

# STEP 5 - computing METRICS scores such as MAP & P@10
start_time = time.time()
print("---------------------------------------------------------------------------------------")
print("")
print("Running The TREC_EVAL to retrieve the MAP & P@10 Scores")

def read_qrel(file_path):
    '''
    read the test file and store it in qrel
    '''
    qrel = {}
    with open(file_path, 'r') as f:
        f.readline() # skip the header
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                query_id, doc_id, relevance = parts
                relevance = int(relevance)
                if query_id not in qrel:
                    qrel[query_id] = {}
                qrel[query_id][doc_id] = relevance
    return qrel

def read_run(file_path):
    '''
    read the results file
    '''
    run = {}
    with open(file_path, 'r') as f:
        for _ in range(80735): # skip the first 80735 lines and start at the line where the first query is 1 - these are the test file queries
            next(f)

        for line in f: # process the rest of the file
            parts = line.strip().split()
            if len(parts) >= 6:
                query_id, _, doc_id, rank, score, _ = parts[:6]
                score = float(score)
                if query_id not in run:
                    run[query_id] = {}
                run[query_id][doc_id] = score

    return run

# get the file paths
qrel_file = "../scifact/qrels/test.tsv"
run_bm25 = "../Results_Scores/BM25/Results.txt"
run_electra = "../Results_Scores/ELECTRA/Results.txt"
run_minilm = "../Results_Scores/MINI_LM/Results.txt"
run_bert = "../Results_Scores/BERT/Results.txt"

# read the files (qrel) and (run)
qrel = read_qrel(qrel_file)
run_bm25 = read_run(run_bm25)
run_electra = read_run(run_electra)
run_minilm = read_run(run_minilm)
run_bert = read_run(run_bert)

# evaluate using pytrec_eval
evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'P_10'}) # get map scores and P_10
results_bm25 = evaluator.evaluate(run_bm25)
results_electra = evaluator.evaluate(run_electra)
results_minilm = evaluator.evaluate(run_minilm)
results_bert = evaluator.evaluate(run_bert)

# save the results to a file
output_file_bm25 = '../Results_Scores/MAP_P10/BM25_MAP_P10_SCORE.json'
output_file_electra = '../Results_Scores/MAP_P10/ELECTRA_MAP_P10_SCORE.json'
output_file_minilm = '../Results_Scores/MAP_P10/MINI_LM_MAP_P10_SCORE.json'
output_file_bert = '../Results_Scores/MAP_P10/BERT_MAP_P10_SCORE.json'

with open(output_file_bm25, 'w') as f:
    json.dump(results_bm25, f, indent=1)

with open(output_file_electra, 'w') as f:
    json.dump(results_electra, f, indent=1)

with open(output_file_minilm, 'w') as f:
    json.dump(results_minilm, f, indent=1)

with open(output_file_bert, 'w') as f:
    json.dump(results_bert, f, indent=1)

# return the path to the results file
print(f"Evaluation results for BM25 model saved to {output_file_bm25}")
print(f"Evaluation results for ELECTRA model saved to {output_file_electra}")
print(f"Evaluation results for MINI LM model saved to {output_file_minilm}")
print(f"Evaluation results for BERT model saved to {output_file_bert}")

# get the average MAP score - average map scores for all queries
total_map_bm25 = sum(results_bm25[query]['map'] for query in results_bm25) / len(results_bm25)
total_map_electra = sum(results_electra[query]['map'] for query in results_electra) / len(results_electra)
total_map_minilm = sum(results_minilm[query]['map'] for query in results_minilm) / len(results_minilm)
total_map_bert = sum(results_bert[query]['map'] for query in results_bert) / len(results_bert)

# round to 4 decimal places
total_map_bm25 = round(total_map_bm25, 4)
total_map_electra = round(total_map_electra, 4)
total_map_minilm = round(total_map_minilm, 4)
total_map_bert = round(total_map_bert, 4)

# print the average map score
print("The average MAP Score for the INITIAL BM25 MODEL is : ", total_map_bm25)
print("The average MAP Score for the ELECTRA MODEL is: ", total_map_electra)
print("The average MAP Score for the MINI LM MODEL is: ", total_map_minilm)
print("The average MAP Score for the BERT MODEL is: ", total_map_bert)

# get the average P@10 score - average P@10 for all queries
total_p10_bm25 = sum(results_bm25[query]['P_10'] for query in results_bm25) / len(results_bm25)
total_p10_electra = sum(results_electra[query]['P_10'] for query in results_electra) / len(results_electra)
total_p10_minilm = sum(results_minilm[query]['P_10'] for query in results_minilm) / len(results_minilm)
total_p10_bert = sum(results_bert[query]['P_10'] for query in results_bert) / len(results_bert)

# round to 4 decimal places
total_p10_bm25 = round(total_p10_bm25, 4)
total_p10_electra = round(total_p10_electra, 4)
total_p10_minilm = round(total_p10_minilm, 4)
total_p10_bert = round(total_p10_bert, 4)

# print the average P@10 score
print("The average P@10 Score for the INITIAL BM25 MODEL is : ", total_p10_bm25)
print("The average P@10 Score for the ELECTRA MODEL is: ", total_p10_electra)
print("The average P@10 Score for the MINI LM MODEL is: ", total_p10_minilm)
print("The average P@10 Score for the BERT MODEL is: ", total_p10_bert)

end_time = time.time()
print(f"\nTime taken to complete STEP 5 - MAP & P@10 COMPUTATION: {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")

# ===============================================================================================================================================

# STEP 6 - finding the top 10 documents for the first 2 queries
start_time = time.time()
def rename_last_column(line, new_name="top_10_documents_first_2_test.tsv_queries_run"):
    line = line.strip()
    if not line:
        return ""
    parts = line.split()
    parts[-1] = new_name
    return " ".join(parts) + "\n"

def compute_top_10_documents_first_2_queries(input_file, output_file, skip_lines=80735):
    with open(input_file, 'r') as f:
        for _ in range(skip_lines): # skip lines to get the test queries
            next(f)
        
        results = [] # store selected lines
        first_query_id = None # hold the query id of the first query block
        first_count = 0 # counter for first query lines
        second_query_id = None # hold the query id of the second query block
        second_count = 0 # counter for second query lines

        for line in f:
            parts = line.strip().split()
            if not parts:
                continue # skip empty lines
            query_id = parts[0] # get the query id which is the first token in each line

            if first_query_id is None: # process first query block
                first_query_id = query_id

            if query_id == first_query_id:
                if first_count < 10:
                    results.append(rename_last_column(line))
                    first_count += 1
                continue  # continue to next query
            
            if second_query_id is None: # process second query block
                second_query_id = query_id  # set second query id
                
            if query_id == second_query_id:
                if second_count < 10:
                    results.append(rename_last_column(line))
                    second_count += 1
                    if second_count == 10: # once 10 lines for second query is reached then it is complete
                        break

            else: # do not need any more queries
                break

    # write results to output file
    with open(output_file, 'w') as out:
        out.writelines(results)
    print(f"Processed {input_file} and wrote results to {output_file}") 

print("---------------------------------------------------------------------------------------")
print("")

print("Computing Top 10 Documents First 2 Queries for the INITIAL BM25 MODEL.")
output_dir = "../Results_Scores/TOP_10_DOCS_FIRST_2_QUERIES"
os.makedirs(output_dir, exist_ok=True)
bm25_source = "../Results_Scores/BM25/Top10AnswersFirst2Queries.txt"
bm25_dest = os.path.join(output_dir, "BM25_TOP_10_DOCS_FIRST_2_QUERIES.txt")
with open(bm25_source, 'r') as f_in, open(bm25_dest, 'w') as f_out:
    for line in f_in:
        f_out.write(rename_last_column(line))
print("Processed ../Results_Scores/BM25/Top10AnswersFirst2Queries.txt and wrote results to ../Results_Scores/TOP_10_DOCS_FIRST_2_QUERIES/BM25_TOP_10_DOCS_FIRST_2_QUERIES.txt")

print("Computing Top 10 Documents First 2 Queries for the ELECTRA MODEL.")
print("Computing Top 10 Documents First 2 Queries for the MINI LM MODEL.")
print("Computing Top 10 Documents First 2 Queries for the BERT MODEL.")
models = ["ELECTRA", "MINI_LM", "BERT"]
for model in models:
    input_path = f"../Results_Scores/{model}/Results.txt"
    output_path = os.path.join(output_dir, f"{model}_TOP_10_DOCS_FIRST_2_QUERIES.txt")
    compute_top_10_documents_first_2_queries(input_path, output_path)
    
end_time = time.time()
print(f"Time taken to complete STEP 6 - TOP 10 DOCUMENTS FIRST 2 QUERIES: {end_time - start_time:.2f} seconds")
print("")
print("---------------------------------------------------------------------------------------")