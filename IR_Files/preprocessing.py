# imports
import nltk
import re
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from utils import progress_bar
import time
import json

def load_stopwords(file_path):
    '''
    retrieve all the stop words
    '''
    with open(file_path, 'r', encoding='utf-8') as file:
        stop_words = {line.strip().lower() for line in file}
    return stop_words

stop_words = load_stopwords('../stopWords.txt')
stemmer = PorterStemmer() # initialize a stemmer

def tokenize(text):
    """
    tokenize the text using NLTK's word_tokenize
    """
    text = re.sub(r'[^a-zA-Z\s]', '', text) # filter out the punctuation and number

    tokens = word_tokenize(text.lower())
    filtered_tokens = []

    for token in tokens:
        if token not in stop_words:
            filtered_tokens.append(token) # filter out stop words

    return filtered_tokens

def stem_tokens(tokens):
    '''
    turns words to their root form by using a stemmer
    '''
    return [stemmer.stem(token) for token in tokens]

def remove_extras(tokens):
    '''
    function for removing extras and unnecessary information
    '''
    return [token for token in tokens if token not in stop_words and token not in ['no_queri', 'no_narr']]

def preprocess_text(text):
    '''
    preprocess the text and return the tokens
    '''
    tokens = tokenize(text)
    tokens = stem_tokens(tokens)
    tokens = remove_extras(tokens)
    return tokens

def preprocess_documents(documents):
    '''
    preprocessing documents for both head and text
    '''
    previousId = "t"
    count = 1
    start_time = time.time()

    for doc in documents:
        fileId = str(doc['DOCNO'].split(" ")[0])
        if (not (fileId == previousId)):
            progress_bar(count, len(documents))
            previousId = fileId
            count = count + 1
        doc['HEAD'] = preprocess_text(doc['HEAD'])
        doc['TEXT'] = preprocess_text(doc['TEXT'])
        
    end_time = time.time()
    print(f"\nTime taken to parse and preprocess documents: {end_time - start_time:.2f} seconds")
    return documents

def preprocess_documents_head_only(documents):
    '''
    preprocessing documents for just the head
    '''
    previousId = "t"
    count = 1
    start_time = time.time()

    for doc in documents:
        fileId = str(doc['DOCNO'].split(" ")[0])
        if (not (fileId == previousId)):
            progress_bar(count, len(documents))
            previousId = fileId
            count = count + 1

        # instead of preproccessing text, remove any doc['TEXT'] and preprocess doc['HEAD']
        doc.pop('TEXT', None)

        # head only
        doc['HEAD'] = preprocess_text(doc['HEAD'])
    
    end_time = time.time()
    print(f"\nTime taken to parse and preprocess documents: {end_time - start_time:.2f} seconds")
    return documents

def preprocess_queries(queries):
    '''
    function for preprocessing queries
    '''
    for query in queries:
        query['title'] = preprocess_text(query['title'])
    return queries

def save_preprocessed_data(data, file_path):
    '''
    save preprocessed data such as documents and queries to a file path
    '''
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)