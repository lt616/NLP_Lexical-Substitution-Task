#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import numpy as np
import string

# Participate in the 4705 lexical substitution competition (optional): NO
# Alias: [please invent some name]

def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos):
    # Part 1
    possible_synonyms = []
    occur_words = {}
    # print(lemma)
    for lexeme in wn.lemmas(lemma, pos=pos):
        for synset_lemma in lexeme.synset().lemmas():
            word = synset_lemma.name()
            word = word.replace("_", " ")

            if word in occur_words or word == lemma:
                continue

            possible_synonyms.append(word)
            occur_words[word] = 1

    # print(possible_synonyms)
    # sys.exit()

    return possible_synonyms

def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    get_candidates(context.lemma, context.pos)

    return 'smurf'

def wn_frequency_predictor(context):
    possible_synonyms = []
    occur_words = {}
    # print("ori " + context.lemma)
    for lexeme in wn.lemmas(context.lemma, pos=context.pos):
        for synset_lemma in lexeme.synset().lemmas():
            word = synset_lemma.name()
            word = word.replace("_", " ")

            if word == context.lemma:
                continue

            if not word in occur_words:
                occur_words[word] = 0
                possible_synonyms.append(word)
            occur_words[word] += synset_lemma.count()

    max = -1
    res = 'smurf'
    for word in occur_words:
        if occur_words[word] > max:
            max = occur_words[word]
            res = word

    return res # replace for part 2

def wn_simple_lesk_predictor(context):
    stop_words_dict = generate_dict(stopwords.words('english'))
    context_dict = generate_dict(context.left_context + [context.lemma] + context.right_context)

    max = -1
    max_synsets = []

    for lexeme in wn.lemmas(context.lemma, pos=context.pos):
        synset_lemma = lexeme.synset()
        # check definition and examples
        occur_count = check_definition_examples(synset_lemma, stop_words_dict, context_dict)

        # check hypernyms
        for hypernyms in synset_lemma.hypernyms():
            occur_count += check_definition_examples(hypernyms, stop_words_dict, context_dict)

        if occur_count > max:
            max = occur_count
            max_synsets = [lexeme]
        elif occur_count == max:
            max_synsets.append(lexeme)

    return lesk_predictor_helper(max_synsets, context)

def lesk_predictor_helper(synsets, context):
    max = -1
    max_synset = None

    for synset in synsets:
        if synset.count() > max:
            max = synset.count()
            max_synset = synset.synset()

    return max_lemma(max_synset, context)

def max_lemma(synset, context):
    max = -1
    res = 'smurf'

    occur_words = {}

    for lemma in synset.lemmas():
        word = lemma.name()
        word = word.replace("_", " ")

        if word == context.lemma:
            continue

        if not word in occur_words:
            occur_words[word] = 0
        occur_words[word] += lemma.count()

    for word in occur_words:
        if occur_words[word] > max:
            max = occur_words[word]
            res = word

    return res

# def non_overlap_predictor(context):
#     max = -1
#     res = 'smurf'
#     max_synset = None

#     for lexeme in wn.lemmas(context.lemma, pos=context.pos):
#         if lexeme.count() > max:
#             max_synset = lexeme.synset()
#             max = lexeme.count()

#     if max_synset is None:
#         return res

#     return max_lemma(max_synset, context)

def check_definition_examples(synset_lemma, stop_words_dict, context_dict):
    occur_count = 0

    # check definition
    for token in tokenize(synset_lemma.definition()):
        if not token in stop_words_dict and token in context_dict:
            occur_count += 1

    # check examples
    for example in synset_lemma.examples():
        for token in tokenize(example):
            if not token in stop_words_dict and token in context_dict:
                occur_count += 1

    return occur_count

def generate_dict(words):
    res = {}

    for word in words:
        res[word] = 1

    return res      
   
class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self, context):

        return None # replace for part 4

    def predict_nearest_with_context(self, context): 
        return None # replace for part 5

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    # W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    # predictor = Word2VecSubst(W2VMODEL_FILENAME)

    # get_candidates('slow','a')

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging

        # prediction = wn_frequency_predictor(context)

        prediction = wn_simple_lesk_predictor(context)

        # prediction = smurf_predictor(context) 
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
