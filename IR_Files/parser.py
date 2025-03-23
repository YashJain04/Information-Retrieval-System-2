import json

def parse_document(document_line):
    """
    Parse a single JSON line as a document
    """
    doc = json.loads(document_line)
    parsed_doc = {
        'DOCNO': doc['_id'],
        'HEAD': doc.get('title', 'NO_TITLE'),
        'TEXT': doc.get('text', 'NO_TEXT'),
    }
    return parsed_doc

def parse_documents_from_file(file_path):
    """
    Read the JSON lines file and parse each document
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        parsed_docs = [parse_document(line) for line in file]
    return parsed_docs

def parse_query(query_line):
    """
    Parse a single JSON line as a query
    """
    query = json.loads(query_line)
    parsed_query = {
        'num': query['_id'],
        'title': query.get('text', 'NO_TEXT'),
    }
    return parsed_query

def parse_queries_from_file(file_path):
    """
    Read the JSON lines file and parse each query
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        parsed_queries = [parse_query(line) for line in file]
    return parsed_queries