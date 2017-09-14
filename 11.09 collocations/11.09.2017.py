import re
from collections import Counter
from nltk.corpus import stopwords
from math import log

with open('rus-wikipedia-sample-companies.txt', encoding='utf-8') as opener:
    text = opener.read().strip().lower()

tokenizer = re.compile('\w+[-]\w+|\w+')
stop_words = stopwords.words('russian')
tokens = tokenizer.findall(text)
total = len(tokens)

unigr_dict = Counter(tokens)
bigr_dict = dict()
last_ind = len(tokens) - 1


for ind, token in enumerate(tokens):
    if ind != last_ind:

        if token not in stop_words:

            if tokens[ind+1] not in stop_words:

                if (token, tokens[ind+1]) in bigr_dict:
                    bigr_dict[(token, tokens[ind+1])] += 1
                else:
                    bigr_dict[(token, tokens[ind+1])] = 1

target_words = ['google']
header = 'word1\tword2\tcount(word1)\tcount(word2)\tcount(bigram)\tpmi\n'

with open('output.tsv', 'w', encoding='utf-8') as writer:
    writer.write(header)

    for target_word in target_words:

        for key, value in bigr_dict.items():

            if target_word in key:

                if value > 10:
                    pmi = log((value/(total-1))/((unigr_dict[key[0]]/total) * (unigr_dict[key[1]]/total)), 2)
                    writer.write(key[0]+'\t'+key[1]+'\t'+str(unigr_dict[key[0]])+'\t'+str(unigr_dict[key[1]])+'\t'+str(value)+'\t'+str(round(pmi, 3))+'\n')
