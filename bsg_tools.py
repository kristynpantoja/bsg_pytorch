import torch
import numpy as np
from nltk.tokenize import RegexpTokenizer

def tokenize_corpus(corpus, tokenizer = RegexpTokenizer(u'(?ui)\\b[a-z]{3,}\\b')):
    assert type(tokenizer) == nltk.tokenize.regexp.RegexpTokenizer, \
      "tokenizer type is not nltk.tokenize.regexp.RegexpTokenizer"
    tokenized_corpus = [tokenizer.tokenize(document.lower()) for document in corpus]
    flattened = [word for document in tokenized_corpus for word in document]
    vocabulary = set(flattened)
    vocabulary = list(vocabulary)
    word2idx = {w: idx + 1 for (idx , w) in enumerate(vocabulary)}
    idx2word = {idx + 1: w for (idx, w) in enumerate(vocabulary)}
    return vocabulary, tokenized_corpus, word2idx, idx2word

def index_document(document, word2idx):
    indexed_document = []
    for word in document:
        indexed_document.append(word2idx.get(word,0))
    return indexed_document

def index_tokenized_corpus(corpus, word2idx):
    return [index_document(document, word2idx) for document in corpus]

def make_document_contexts(document, window):
    contexts = []
    for context in range(window, len(document) - window):
        center_word = document[context]
        context_words = np.concatenate((document[(context - window) : context],
                                        document[(context + 1) : (context + window + 1)]))
        contexts.append((center_word, context_words))
    return contexts

def get_corpus_centers_contexts(indexed_corpus, window):
    '''
    input an indexed corpus
    output is tuples of all center words and corresponding context words in corpus
    '''
    padded_corpus = [np.pad(document, (window,window), 'constant', constant_values=(0, 0)) \
        for document in indexed_corpus]
    corpus_document_contexts = [make_document_contexts(document, window) for document in padded_corpus]
    # flatten the list - we only care about all the contexts, irrespective of document, for now
    centers_contexts_tuples = [pair for document_contexts in corpus_document_contexts for pair in document_contexts]
    centers_list_contexts_list = list(map(list, zip(*centers_contexts_tuples)))
    corpus_center_words = torch.tensor(centers_list_contexts_list[0], dtype = torch.int64)
    corpus_context_words = torch.tensor(centers_list_contexts_list[1], dtype = torch.int64)
    return corpus_center_words, corpus_context_words

def make_unigram_dict(corpus_center_words):
    tokens = corpus_center_words.tolist()
    types, counts = np.unique(corpus_center_words, return_counts=True)
    unigram_dict = dict(zip(types, counts/len(tokens)))
    return unigram_dict
