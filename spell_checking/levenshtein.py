import Levenshtein
import time
from sklearn.externals import joblib
import math
import re

start_time = time.time()

digits_search = re.compile('\d')

word_set = {}

with open('1grams-3.txt', encoding='utf-8') as opener:
    for line in opener:
        freq, word = line.split('\t')
        word = word.rstrip().lower()
        if word in word_set:
            word_set[word] += int(freq)
        else:
            word_set[word] = int(freq)

"""bigrams = {}
with open('2grams-3.txt', encoding='utf-8') as bigram_opener:
    #lines = bigram_opener.readlines()
    for line in bigram_opener:
        table = line.split('\t')
        word_1 = table[1].rstrip().lower()
        word_2 = table[-1].rstrip().lower()
        if (word_1, word_2) in bigrams:
            bigrams[(word_1, word_2)] += int(table[0])
        else:
            bigrams[(word_1, word_2)] = int(table[0])"""

# just to see how it works
test_strings = []
with open('test_sample_testset.txt', encoding='utf-8') as strings_opener:
    for i, line in enumerate(strings_opener):
        if i > 110 and i < 120:
            test_strings.append(line.strip())

print("Elapsed time for loading: {:.3f} sec".format(time.time() - start_time))

def spell_checking (string):
    corrected = []  # result of spell_checking
    tokens = string.split(' ')

    for ind, token in enumerate(tokens):
        candidates = []
        candidates_ed = []

        # if a token is in our set, let it be
        if (token in word_set) or (len(token) < 4):
                corrected.append(token)

        # if a token consists of digits only, let it be
        elif digits_search.search(token):
            counter = 0

            for char in token:
                if digits_search.search(char):
                    counter += 1
            #checking if there are digits only in this token
            if counter == len(token):
                corrected.append(token)

        else:
            for example in word_set.keys():
                absolute = abs(len(token) - len(example))
                example_ed = Levenshtein.distance(token, example)/(len(token) + math.log10(len(token) + absolute))
                if example_ed <= 2/len(token) + math.log10(len(token) + absolute):
                    candidates.append(example)
                    candidates_ed.append(example_ed)

            minimum = min(candidates_ed)
            candidates = [candidates[i] for i in range(len(candidates)) if candidates_ed[i] == minimum]
            corrected.append(candidates[0]) # just pick the first one; yes, it's a dumb way, but nevertheless..

            # an attempt to sort the candidates after computing levenshtein distance by using jaro_winkler
            """jaro_distance_list = []

            for word in candidates:
                jaro_distance = 1 - Levenshtein.jaro_winkler(token, word)
                jaro_distance_list.append(jaro_distance)

            minimum_jaro = min(jaro_distance_list)
            corrected.append([candidates[i] for i in range(len(candidates)) if jaro_distance_list[i] == minimum_jaro])"""

            # an attempt to sort the candidates after computing levenshteon distance by using bigram freqs
            """ candidates_bigr_freqs = []
            for candidate in candidates:
                bigrams_freq = 0
                if ind == 0:
                    if (candidate, tokens[ind+1]) in bigrams:
                        bigrams_freq += bigrams[(candidate, tokens[ind+1])]
                elif ind == string_len-1:
                    if (corrected[-1], candidate) in bigrams:
                        bigrams_freq += bigrams[(corrected[-1], candidate)]
                else:
                    if (corrected[-1], candidate) in bigrams:
                        bigrams_freq += bigrams[(corrected[-1], candidate)]
                    if (candidate, tokens[ind+1]) in bigrams:
                        bigrams_freq += bigrams[(candidate, tokens[ind+1])]
                candidates_bigr_freqs.append(bigrams_freq)

            optimal_candidate = candidates[candidates_bigr_freqs.index(max(candidates_bigr_freqs))]
            #print(token)
            #print(candidates)
            #print(candidates_bigr_freqs)
            corrected.append(optimal_candidate)

            max_freq = 0
            optimal_candidate = ''
            for candidate in candidates:
                cand_freq = word_set[candidate]
                if cand_freq > max_freq:
                    max_freq = cand_freq
                    optimal_candidate = candidate

            corrected.append(optimal_candidate)"""

    print(string)
    return (corrected)

for test_string in test_strings:
    print(spell_checking(test_string))
    print("Elapsed time for searching: {:.3f} sec".format(time.time() - start_time))