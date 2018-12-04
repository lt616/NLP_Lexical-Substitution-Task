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

# Function best_predictor() is my own solution, it is based on part 03 implementation. 
# I improved part03 with the help of Morphy.
def best_predictor(context):
    stop_words_dict = generate_dict(stopwords.words('english'))
    context_with_morphy = add_morphy(context.left_context + [context.lemma] + context.right_context)
    context_dict = generate_dict(context_with_morphy)

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

def add_morphy(context_arr):
    context_morphy = []
    
    for lemma in context_arr:
        lemma_morphy = wn.morphy(lemma)
        if not lemma_morphy is None and not lemma_morphy == lemma:
            context_morphy.append(lemma_morphy)

    return context_arr + context_morphy


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
        possible_synonyms = get_candidates(context.lemma, context.pos)

        max = -1
        res = 'smurf'
        for synonyms in possible_synonyms:
            try:
                similarity = self.model.similarity(context.lemma, synonyms)
                if similarity > max:
                    max = similarity
                    res = synonyms
            except:
                continue

        return res

    def predict_nearest_with_context(self, context): 
        stop_words_dict = generate_dict(stopwords.words('english'))

        left_context = self.remove_stop_words(stop_words_dict, context.left_context)
        right_context = self.remove_stop_words(stop_words_dict, context.right_context)

        # print(left_context)
        # print(right_context)

        left_context_limit = self.limit_left_context_len(left_context)
        right_context_limit = self.limit_right_context_len(right_context)

        # print(left_context_limit)
        # print(right_context_limit)

        # Sum up sentence vector
        vector = self.model.wv[context.lemma]
        for lemma in left_context_limit:
            try:
                vector = np.add(vector, self.model.wv[lemma])
            except:
                continue
        for lemma in right_context_limit:
            try:
                vector = np.add(vector, self.model.wv[lemma])
            except:
                continue
        possible_synonyms = get_candidates(context.lemma, context.pos)

        max = -1
        res = 'smurf'

        for synonyms in possible_synonyms:
            try:
                similarity = cos(self.model.wv[synonyms], vector)
                if similarity > max:
                    max = similarity
                    res = synonyms
            except:
                continue

        return res

    def remove_stop_words(self, stop_words_dict, context_arr):
        res = []

        for lemma in context_arr:
            if not lemma in stop_words_dict:
                res.append(lemma)

        return res

    def limit_left_context_len(self, context_arr):
        if len(context_arr) == 0:
            return context_arr

        size = 5
        if len(context_arr) < 5:
            size = len(context_arr)

        return context_arr[-size:]

    def limit_right_context_len(self, context_arr):
        if len(context_arr) == 0:
            return context_arr

        size = 5
        if len(context_arr) < 5:
            size = len(context_arr)

        return context_arr[:size]

def cos(v1,v2):
    return np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))


if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    # W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    print("loading..")
    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    # get_candidates('slow','a')

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging

        prediction = predictor.predict_nearest_with_context(context)

        # prediction = predictor.predict_nearest(context)

        # prediction = wn_frequency_predictor(context)

        # prediction = wn_simple_lesk_predictor(context)

        # print(get_candidates(context.lemma, context.pos))

        # prediction = smurf_predictor(context) 
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
