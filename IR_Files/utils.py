# imports
import sys
import json
from ranking import normalize_scores

def progress_bar(current, total, bar_length=50):
    '''
    creating a progress bar to help user see how much completed and remaining in each step of the process
    '''
    progress = current / total
    block = int(bar_length * progress)
    bar = '█' * block + '-' * (bar_length - block)
    percent = progress * 100
    text = f"\r{percent:.2f}%|{bar}| {current}/{total}"
    sys.stdout.write(text)
    sys.stdout.flush()

def writeResults(results_file, queries, bm25, run_name):
    '''
    only the top documents for each query
    '''
    beir_results = {} # map each query_id to a dictionary of doc_id and score.
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
                    beir_results[query_id] = [(top_doc_id, top_score)] # build the per-query result as a dict mapping doc_id to score
                else:
                    result_line = f"{query_id} Q0 {top_doc_id} 1 {top_score} {run_name}\n"
                    output_file.write(result_line)

        if 'json' in results_file:
            json.dump(beir_results, output_file, indent=4)

def writeResultsTop10First2(results_file, queries, bm25, run_name):
    '''
    only the top 10 documents for each query
    '''
    beir_results = {} # map each query_id to a dictionary of doc_id and score.
    processed = 0  # keep track of how many processed so far
    started = False  # flag indicating that query with ID 1 has been encountered

    with open(results_file, 'w') as output_file:
        for idx, query in enumerate(queries):
            query_id = query['num']
            if not started: # wait until a query ID with 1 is found (start of test queries)
                if query_id == "1" or query_id == 1:
                    started = True
                else:
                    continue  # skip any queries until ID 1 is reached

            # process the query if already have started and haven't processed two queries yet
            if started and processed < 2:
                query_terms = query['title']
                progress_bar(idx + 1, len(queries))
                ranked_docs = bm25.rank_documents(query_terms)
                normalized_ranked_docs = normalize_scores(ranked_docs)
                normalized_ranked_docs = normalized_ranked_docs[:10] # keep only the top 10 results
                processed += 1

                if 'json' in results_file:
                    beir_results[query_id] = [(doc_id, score) for doc_id, score in normalized_ranked_docs] # build the per-query result as a dict mapping doc_id to score
                else:
                    for rank, (doc_id, score) in enumerate(normalized_ranked_docs, start=1):
                        result_line = f"{query_id} Q0 {doc_id} {rank} {score} {run_name}\n"
                        output_file.write(result_line)

                if processed == 2: # stop processing after first 2 queries done
                    break

        if 'json' in results_file:
            json.dump(beir_results, output_file, indent=4)

def writeResultsAll(results_file, queries, bm25, run_name):
    '''
    all the documents for each query
    '''
    beir_results = {} # map each query_id to a dictionary of doc_id and score.
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
                beir_results[query_id] = [(doc_id, score) for doc_id, score in normalized_ranked_docs] # build the per-query result as a dict mapping doc_id to score
            
            else:
                for rank, (doc_id, score) in enumerate(normalized_ranked_docs, start=1):
                    result_line = f"{query_id} Q0 {doc_id} {rank} {score} {run_name}\n"
                    output_file.write(result_line)

        if ('json' in results_file):
            json.dump(beir_results, output_file, indent=4)

def writeResultsTop100(results_file, queries, bm25, run_name):
    '''
    only the top 100 documents for each query - this is the main results file to be used
    '''
    beir_results = {} # map each query_id to a dictionary of doc_id and score.
    count = 1

    with open(results_file, 'w') as output_file:
        for query in queries:
            query_id = query['num']
            if isinstance(query['title'], list):
                query_terms = query['title']
            else:
                query_terms = query['title'].split()
                
            progress_bar(count, len(queries))
            ranked_docs = bm25.rank_documents(query_terms)
            normalized_ranked_docs = normalize_scores(ranked_docs)
            count += 1

            normalized_ranked_docs = normalized_ranked_docs[:100] # keep only the top 100 results

            beir_results[query_id] = {doc_id: score for doc_id, score in normalized_ranked_docs} # build the per-query result as a dict mapping doc_id to score

            for rank, (doc_id, score) in enumerate(normalized_ranked_docs, start=1):
                result_line = f"{query_id} Q0 {doc_id} {rank} {score} {run_name}\n"
                output_file.write(result_line)

        if 'json' in results_file:
            json.dump(beir_results, output_file, indent=4)

    return beir_results