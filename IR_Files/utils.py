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
    count = 1

    with open(results_file, 'w') as output_file:
        for query in queries:
            query_id = query['num']
            query_terms = query['title']
            progress_bar(count, len(queries))
            ranked_docs = bm25.rank_documents(query_terms)
            normalized_ranked_docs = normalize_scores(ranked_docs)
            count += 1

            # keep only top 10 for the first two queries
            if count < 4:
                normalized_ranked_docs = normalized_ranked_docs[:10]

                if 'json' in results_file:
                    beir_results[query_id] = [(doc_id, score) for doc_id, score in normalized_ranked_docs]
                
                else:
                    for rank, (doc_id, score) in enumerate(normalized_ranked_docs, start=1):
                        result_line = f"{query_id} Q0 {doc_id} {rank} {score} {run_name}\n"
                        output_file.write(result_line)

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

            # get the top 100
            normalized_ranked_docs = normalized_ranked_docs[:100]

            if ('json' in results_file):
                beir_results[query_id] = [(doc_id, score) for doc_id, score in normalized_ranked_docs]
            
            else:
                for rank, (doc_id, score) in enumerate(normalized_ranked_docs, start=1):
                    result_line = f"{query_id} Q0 {doc_id} {rank} {score} {run_name}\n"
                    output_file.write(result_line)

        if ('json' in results_file):
            json.dump(beir_results, output_file, indent=4)