import sys
import json
from ranking import normalize_scores

def progress_bar(current, total, bar_length=50):
    progress = current / total
    block = int(bar_length * progress)
    bar = '█' * block + '-' * (bar_length - block)
    percent = progress * 100
    text = f"\r{percent:.2f}%|{bar}| {current}/{total}"
    sys.stdout.write(text)
    sys.stdout.flush()

def writeResults(results_file, queries, bm25, run_name):
    beir_results = {}
    count = 1

    with open(results_file, 'w') as output_file:
        for query in queries:
            query_id = query['num']
            query_terms = query['title']
            progress_bar(count, len(queries))
            ranked_docs = bm25.rank_documents(query_terms)
            normalized_ranked_docs = normalize_scores(ranked_docs)
            count += 1

            if normalized_ranked_docs:  # ensure there's at least one result
                top_doc_id, top_score = normalized_ranked_docs[0]  # get only the top result

                if 'json' in results_file:
                    beir_results[query_id] = [(top_doc_id, top_score)]
                else:
                    result_line = f"{query_id} Q0 {top_doc_id} 1 {top_score} {run_name}\n"
                    output_file.write(result_line)

        if 'json' in results_file:
            json.dump(beir_results, output_file, indent=4)

def writeResultsTop10First2(results_file, queries, bm25, run_name):
    beir_results = {}
    processed = 0  # Number of queries processed
    started = False  # Flag indicating that we encountered query with ID 1

    with open(results_file, 'w') as output_file:
        for idx, query in enumerate(queries):
            query_id = query['num']
            # Wait until we find the query with ID 1.
            if not started:
                # the query ID with 1 is where the test queries start
                if query_id == "1" or query_id == 1:
                    started = True
                else:
                    continue  # Skip any queries until ID 1 is reached.

            # Process the query if we have started and haven't processed two queries yet.
            if started and processed < 2:
                query_terms = query['title']
                progress_bar(idx + 1, len(queries))
                ranked_docs = bm25.rank_documents(query_terms)
                normalized_ranked_docs = normalize_scores(ranked_docs)
                # Keep only the top 10 results.
                normalized_ranked_docs = normalized_ranked_docs[:10]
                processed += 1

                if 'json' in results_file:
                    beir_results[query_id] = [(doc_id, score) for doc_id, score in normalized_ranked_docs]
                else:
                    for rank, (doc_id, score) in enumerate(normalized_ranked_docs, start=1):
                        result_line = f"{query_id} Q0 {doc_id} {rank} {score} {run_name}\n"
                        output_file.write(result_line)

                # Once we've processed two queries, stop processing further.
                if processed == 2:
                    break

        if 'json' in results_file:
            json.dump(beir_results, output_file, indent=4)

def writeResultsAll(results_file, queries, bm25, run_name):
    beir_results = {}
    count = 1

    with open(results_file, 'w') as output_file:
        for query in queries:
            query_id = query['num']
            query_terms = query['title']
            progress_bar(count, len(queries))
            ranked_docs = bm25.rank_documents(query_terms)
            normalized_ranked_docs = normalize_scores(ranked_docs)
            count += 1

            # get all
            if ('json' in results_file):
                beir_results[query_id] = [(doc_id, score) for doc_id, score in normalized_ranked_docs]
            
            else:
                for rank, (doc_id, score) in enumerate(normalized_ranked_docs, start=1):
                    result_line = f"{query_id} Q0 {doc_id} {rank} {score} {run_name}\n"
                    output_file.write(result_line)

        if ('json' in results_file):
            json.dump(beir_results, output_file, indent=4)

def writeResultsTop100(results_file, queries, bm25, run_name):
    # This dictionary will map each query_id to a dictionary of doc_id and score.
    beir_results = {}
    count = 1

    # Open the file for writing output results.
    with open(results_file, 'w') as output_file:
        for query in queries:
            query_id = query['num']
            # For BM25, the ranking function typically expects a tokenized query.
            # If your query['title'] is already a list, use it directly;
            # otherwise, you can tokenize (e.g., by splitting) or use the tokens you computed earlier.
            if isinstance(query['title'], list):
                query_terms = query['title']
            else:
                query_terms = query['title'].split()
                
            progress_bar(count, len(queries))
            ranked_docs = bm25.rank_documents(query_terms)
            normalized_ranked_docs = normalize_scores(ranked_docs)
            count += 1

            # Get the top 100 results.
            normalized_ranked_docs = normalized_ranked_docs[:100]

            # Build the per-query result as a dict mapping doc_id to score.
            beir_results[query_id] = {doc_id: score for doc_id, score in normalized_ranked_docs}

            # Also write the results to file in the expected format.
            for rank, (doc_id, score) in enumerate(normalized_ranked_docs, start=1):
                result_line = f"{query_id} Q0 {doc_id} {rank} {score} {run_name}\n"
                output_file.write(result_line)

        # Optionally, if the results file is supposed to be JSON, dump the dict.
        if 'json' in results_file:
            json.dump(beir_results, output_file, indent=4)

    # Return the dictionary so that it can be used for reranking.
    return beir_results