import Levenshtein
import re
import time
from sklearn.externals import joblib
import pymorphy2

morph_analyzer = pymorphy2.MorphAnalyzer()
start_time = time.time()
keyboard_lines = ['йцукенгшщзхъ', 'фывапролджэ', 'ячсмитьбю']
neighbors = {}

# making dictionary for misspells with higher probabilities
for ind_row, row in enumerate(keyboard_lines):
    for ind_char, char in enumerate(row):
        if ind_row == 0:
            neighbors[char] = [keyboard_lines[ind_row][ind_char - 1:ind_char],
                               keyboard_lines[ind_row][ind_char + 1:ind_char + 2],
                               keyboard_lines[ind_row + 1][ind_char:ind_char + 1]]

        elif ind_row == 1:
            neighbors[char] = [keyboard_lines[ind_row - 1][ind_char:ind_char + 1], keyboard_lines[ind_row][ind_char - 1:ind_char],
                               keyboard_lines[ind_row][ind_char + 1:ind_char + 2],
                               keyboard_lines[ind_row + 1][ind_char:ind_char + 1]]

        elif ind_row == 2:
            neighbors[char] = [keyboard_lines[ind_row - 1][ind_char:ind_char + 1],
                                keyboard_lines[ind_row][ind_char - 1:ind_char],
                               keyboard_lines[ind_row][ind_char + 1:ind_char + 2]]

#loading our word_set of candidates
"""word_set = joblib.load('all_words.pkl')
print("Elapsed time for loading: {:.3f} sec".format(time.time() - start_time))"""

with open('1grams-3.txt', encoding='utf-8') as opener:
    lines = opener.readlines()

word_set = {line.split('\t')[1].rstrip() for line in lines}
print("Elapsed time for loading: {:.3f} sec".format(time.time() - start_time))

# cleaning up the data (lists' cuts from indices out of range return empty strings, so it's urgent to get rid of them)
for key, value in neighbors.items():
    if '' in value:
        value = set(value)
        value.remove('')
    print(key, value)

# iterative with two matrix rows
def levenshtein(target_word, candidate):

    if target_word == candidate:
        return 0

    elif len(target_word) == 0:
        return len(candidate)

    elif len(candidate) == 0:
        return len(target_word)
    v0 = [None] * (len(candidate) + 1)
    v1 = [None] * (len(candidate) + 1)

    for i in range(len(v0)):
        v0[i] = i

    for i in range(len(target_word)):
        v1[0] = i + 1

        for j in range(len(candidate)):
            #cost = 0 if s[i] == t[j] else 1
            if target_word[i] == candidate[j]:
                cost = 0
            elif candidate[j] in neighbors[target_word[i]]:
                cost = 0.5
            else:
                cost = 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)

        for j in range(len(v0)):
            v0[j] = v1[j]
    return v1[len(candidate)] / len(candidate)

digits_search = re.compile('\d')

def spell_checking (string):
    corrected = []  # result of spell_checking
    tokens = string.split(' ')

    for token in tokens:
        candidates = []
        candidates_ed = []

        # if a token is in our set, let it be
        if token in word_set:
                corrected.append(token)

        # if a token consists of digits only, let it be
        elif digits_search.search(token):
            counter = 0

            for char in token:
                if digits_search.search(char):
                    counter += 1
            #checking if there are only digits in this token
            if counter == len(token):
                corrected.append(token)
        else:
            for example in word_set:
                example_ed = levenshtein(token, example)
                # we've chosen 2/len(token) as a margin for candidates because of a supposition that usually people make one-twp mistakes
                # while misspelling a word (or making one mistake changing the order of the chars)
                if example_ed <= 2/len(token):
                    candidates.append(example)
                    candidates_ed.append(example_ed)

            minimum = min(candidates_ed)
            candidates = [(candidates[i], candidates_ed[i]) for i in range(len(candidates)) if candidates_ed[i] == minimum]
            corrected.append(candidates)
        print("Elapsed time for a word: {:.3f} sec".format(time.time() - start_time))
    return (string, corrected)

test_string = 'схездил птом в ркб'
print(spell_checking(test_string))