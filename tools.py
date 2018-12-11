import torch
import numpy as np

import nltk
from nltk.tokenize import RegexpTokenizer

def flatten(list_of_lists):
    return [element for list in list_of_lists for element in list]

def tokenize_corpus(corpus, tokenizer = RegexpTokenizer(u'(?ui)\\b[a-z]{3,}\\b')):
    assert type(tokenizer) == nltk.tokenize.regexp.RegexpTokenizer, \
      "tokenizer type is not nltk.tokenize.regexp.RegexpTokenizer"
    tokenized_corpus = [tokenizer.tokenize(document.lower()) for document in corpus]
    flattened = flatten(tokenized_corpus)
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

def make_unigram_dict(token_indices):
    '''
    is expecting a list
    '''
    # compute unigram frequencies
    types, counts = np.unique(token_indices, return_counts=True)
    unigram_dict = dict(zip(types, counts/len(token_indices)))
    return unigram_dict

def subsample(word_frequency, t = np.power(10.0,-5.0)):
    prob_word = np.maximum(0.0, 1.0 - t/word_frequency)
    return np.random.binomial(size=prob_word.shape[0], n=1, p= prob_word) == 1

def subsample_corpus(indexed_corpus, unigram, t = np.power(10.0,-5.0)):
    subsampled_corpus = []
    for document in indexed_corpus:
        word_freqs = [unigram.get(index) for index in document]
        is_subsampled = subsample(word_freqs, t)
        subsampled_corpus.append(list(np.array(document)[is_subsampled]))
    return subsampled_corpus

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
    centers_contexts_tuples = flatten(corpus_document_contexts)
    centers_list_contexts_list = list(map(list, zip(*centers_contexts_tuples)))
    corpus_center_words = torch.tensor(centers_list_contexts_list[0], dtype = torch.int64)
    corpus_context_words = torch.tensor(centers_list_contexts_list[1], dtype = torch.int64)
    return corpus_center_words, corpus_context_words
