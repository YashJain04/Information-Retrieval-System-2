
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
#     '''
#     Read the results file (Results.txt) and store it in run,
#     skipping the first 80,735 lines to get the test queries.
#     '''
#     run = {}
#     with open(file_path, 'r') as f:
#         # skip the first 80,735 lines and start at the line where the first query is 1 (these are the test.tsv queries)
#         for _ in range(80735):
#             next(f)
        
#         # process the rest of the file
#         for line in f:
#             parts = line.strip().split()  # split by any whitespace
#             if len(parts) >= 6:  # ensure we have at least 6 columns
#                 query_id, _, doc_id, rank, score, _ = parts[:6]
#                 score = float(score)
#                 if query_id not in run:
#                     run[query_id] = {}
#                 run[query_id][doc_id] = score

#     return run

# # get the file paths
# qrel_file = "../scifact/qrels/test.tsv"
# run_file = '../Results_Scores/Results.txt'

# # read the files (qrel) and (run)
# qrel = read_qrel(qrel_file)
# run = read_run(run_file)

# # evaluate using pytrec_eval
# evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg'})
# results = evaluator.evaluate(run)

# # save the results to a file
# output_file = '../Results_Scores/MAPScores.json'
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

# ASSIGNMENT 2 - NEURAL RANKING

# start the timer
start_time = time.time()

# init models
model_type = "BERT"
model_name = "msmarco-distilbert-base-v3"

# call neural results
neural_results = neural_rank_documents(model_type, model_name, documents, inverted_index, doc_lengths, queries, False)

# end the timer
end_time = time.time()

# create the results file
results_file = '../Results_Scores/NeuralScores.txt'

# save the results
neural_save_results(results, results_file)

# print results
print(f"\nTime taken to rank documents: {end_time - start_time:.2f} seconds")
print(f"Ranking results written to {results_file}")
