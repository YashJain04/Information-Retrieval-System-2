import time
import json
# Keep the imports that parse and preprocess:
from parser import parse_documents_from_file, parse_queries_from_file
from preprocessing import preprocess_documents, preprocess_queries, save_preprocessed_data
# Remove or ignore the indexing imports that you no longer need
# from indexing import build_inverted_index, calculate_document_lengths, save_inverted_index
# from ranking import BM25   # Not needed anymore
from utils import writeResults, writeResultsTop10First2, writeResultsAll, writeResultsTop100

# -------------------
# 1) IMPORT BERT_Ranker
# -------------------
from bert_ranking import BERT_Ranker, normalize_scores
# Or wherever you put your BERT ranker class

# File paths
doc_folder_path = '../scifact/corpus.jsonl'
query_file_path = '../scifact/queries.jsonl'
preprocessed_docs_path = '../Results_Scores/preprocessed_documents.json'
preprocessed_queries_path = '../Results_Scores/preprocessed_queries.json'

# -----------------------------------------------------------
# STEP 0 - Parse Queries (and/or docs if needed)
# -----------------------------------------------------------
start_time = time.time()
print("Parsing queries")
queries = parse_queries_from_file(query_file_path)
end_time = time.time()
print(f"Time (STEP 0): {end_time - start_time:.2f} seconds\n")

# -----------------------------------------------------------
# STEP 1 - Preprocess documents and queries
# -----------------------------------------------------------
print("Preprocessing documents & queries")
start_time = time.time()
documents = parse_documents_from_file(doc_folder_path)
documents = preprocess_documents(documents)
save_preprocessed_data(documents, preprocessed_docs_path)

queries = preprocess_queries(queries)
save_preprocessed_data(queries, preprocessed_queries_path)
end_time = time.time()
print(f"Time (STEP 1): {end_time - start_time:.2f} seconds\n")

# -----------------------------------------------------------
# STEP 2 - Build or load BERT embeddings
# (Previously: build_inverted_index for BM25)
# -----------------------------------------------------------
print("Loading documents into dictionary for BERT ranker")

# 1) Convert the preprocessed documents into a dict { doc_id -> string_of_all_tokens }
docs_dict = {}
for doc in documents:
    doc_id = doc["DOCNO"]
    # For BERT, we typically need the text as a single string:
    combined_text = " ".join(doc["HEAD"]) + " " + " ".join(doc["TEXT"])
    docs_dict[doc_id] = combined_text

# 2) Create a BERT ranker (this will embed the docs)
start_time = time.time()
bert_ranker = BERT_Ranker(docs_dict, model_name='sentence-transformers/all-MiniLM-L6-v2')
end_time = time.time()
print(f"Time (STEP 2, building BERT embeddings): {end_time - start_time:.2f} seconds\n")

# -----------------------------------------------------------
# STEP 3 - Retrieval/Ranking
# -----------------------------------------------------------
print("Ranking documents using BERT")
start_time = time.time()

# Use the same utilities you used for BM25, but pass bert_ranker instead.
# Note: The 'writeResults' function calls ranker.rank_documents(query_terms)
#       So we must ensure BERT_Ranker has a rank_documents(...) method
#       that returns [(doc_id, score), ...].

writeResults("../Results_Scores/TopScoresAllQueries_BERT.txt", queries, bert_ranker, "BERT_run")
writeResultsTop10First2("../Results_Scores/Top10AnswersFirst2Queries_BERT.txt", queries, bert_ranker, "BERT_run")
writeResultsAll("../Results_Scores/AllScoresAllQueries_BERT.txt", queries, bert_ranker, "BERT_run")
writeResultsTop100("../Results_Scores/Results_BERT.txt", queries, bert_ranker, "BERT_run")

end_time = time.time()
print(f"Time (STEP 3, Ranking): {end_time - start_time:.2f} seconds\n")

# -----------------------------------------------------------
# STEP 4 - (Optional) Evaluate with pytrec_eval
# -----------------------------------------------------------
import pytrec_eval

def read_qrel(file_path):
    qrel = {}
    with open(file_path, 'r') as f:
        header = f.readline()  # skip the header
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                query_id, doc_id, relevance = parts
                if query_id not in qrel:
                    qrel[query_id] = {}
                qrel[query_id][doc_id] = int(relevance)
    return qrel

def read_run(file_path):
    run = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                query_id, _, doc_id, rank, score, _ = parts[:6]
                if query_id not in run:
                    run[query_id] = {}
                run[query_id][doc_id] = float(score)
    return run

print("Evaluating the BERT results with pytrec_eval")

qrel_file = "../scifact/qrels/test.tsv"
run_file = "../Results_Scores/Results_BERT.txt"  # Our new BERT results
qrel = read_qrel(qrel_file)
run = read_run(run_file)

evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg', 'P.10'})
results = evaluator.evaluate(run)

# Calculate average metrics
map_scores = [v['map'] for v in results.values()]
ndcg_scores = [v['ndcg'] for v in results.values()]
p10_scores  = [v['P.10'] for v in results.values()]
avg_map  = sum(map_scores)/len(map_scores)
avg_ndcg = sum(ndcg_scores)/len(ndcg_scores)
avg_p10  = sum(p10_scores)/len(p10_scores)

print(f"MAP:  {avg_map:.4f}")
print(f"NDCG: {avg_ndcg:.4f}")
print(f"P@10: {avg_p10:.4f}")