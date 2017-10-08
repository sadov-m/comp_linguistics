import string
from wordsegment import segment, load

load()
"""
with open('SCL-OPP/SCL-OPP.txt', encoding='utf-8') as open_scl_opp:
    data_scl = [elem.split('\t') for elem in open_scl_opp.read().split('\n')]
    lexicon = [entry[0].lower() for entry in data_scl]
    polarity = [float(entry[1]) for entry in data_scl]
    """

all_punct_en = set(string.punctuation)
