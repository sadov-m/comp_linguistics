import codecs
import collections
import time
import numpy as np

start_time = time.time()
# reading corpora and dividing it in parts


def read_ruscorpora(filename):
    sent = [[]]
    with codecs.open(filename, encoding='utf-8') as inp_file:
        for line in inp_file:
            line = line.strip('\r\n ')
            if '\t' in line:
                line = line[:line.find('\t')]
            if line.startswith('#'):
                continue
            if not line:
                sent.append([])
                continue
            wordform, lemma, gram = line.rsplit('/', 2)
            pos = gram[:gram.find('=')] if '=' in gram else gram
            pos = pos.split(',')[0]
            sent[-1].append((wordform.lower(), pos, lemma.lower(), gram))
    return sent

# corpus = list of sents, corpus[i] = list of word-tuples(word, tag, lemma, grammemes)
corpus = read_ruscorpora('ruscorpora.parsed.txt')
print("Elapsed time for reading: {:.3f} sec".format(time.time() - start_time))

counts_pos_pos = collections.defaultdict(int)
counts_word_pos = collections.defaultdict(int)
counts_pos = collections.defaultdict(int)

# B - beginning, E - end
BOS = u'<s>'
EOS = u'</s>'

# counting freqs for pos tags, for transitions and for emitions
for sent in corpus:
    if len(sent) == 0:
        continue
    counts_pos[BOS] += 1
    items = [[BOS, BOS]] + sent + [[EOS, EOS]]
    for i, word in enumerate(items[1:]):
        prev_tag = items[i][1]
        word_pos = (word[0], word[1])

        pos_pos = (prev_tag, word[1])

        if pos_pos == (BOS, EOS):
            print(items)

        counts_pos[word[1]] += 1  # tag of word
        counts_pos_pos[pos_pos] += 1  # transitions
        counts_word_pos[word_pos] += 1  # emitions

trans_prob = {key: float(counts_pos_pos[key]) / counts_pos[key[0]] for key in counts_pos_pos}  # transition prob-s
word_prob = {key: float(counts_word_pos[key]) / counts_pos[key[1]] for key in counts_word_pos}  # emission prob-s

# for transitions
def get_trans_prob(tags, trans_probs):
    return -np.log(trans_probs[tags]) if tags in trans_probs else np.inf

# for emissions
def get_word_prob(word_tag_tuple, word_probs, n_tags):
    coeff = 0.99
    return -np.log((coeff * word_probs[word_tag_tuple] if word_tag_tuple in word_probs else 0.0) + (1 - coeff) * 1.0 / n_tags)

words = [item[0] for item in corpus[1]]
tags = list(counts_pos.keys())
number_of_tags = len(tags)
print(tags, type(tags))

viterbi = np.zeros((len(words), len(tags)))
back_point = np.zeros((len(words), len(tags)))

# viterbi[i, :] for words[i] - calculating tag prob for this word
# AND HERE COMES THE ALGORITHM!

states = []
for i in range(len(words)):
    if i == 0:
        viterbi[i, :] = [get_trans_prob((BOS, tag), trans_prob) + get_word_prob((words[i], tag), word_prob, len(tags)) for
                     tag in tags]
        back_point[i, :] = [0 for i in range(len(tags))]
    else:
        viterbi_step = []
        back_point_step = []

        for tag in tags:
            back_point_refs = [get_trans_prob((prev_tag, tag), trans_prob) + viterbi[i-1][ind] for ind, prev_tag in enumerate(tags)]
            minimum = min(back_point_refs)
            back_point_step.append(back_point_refs.index(minimum))  # changed!!!
            viterbi_step.append(minimum + get_word_prob((words[i], tag), word_prob, number_of_tags))

        viterbi[i, :] = viterbi_step
        back_point[i, :] = back_point_step

    if i == len(words)-1:

        for j in range(len(words)-1, -1, -1):

            if j == len(words)-1:

                ind_for_back_point = list(viterbi[j]).index(min(viterbi[j]))
                states.append(tags[ind_for_back_point])
                states.append(tags[int(back_point[j][ind_for_back_point])])

            elif j > 0 and j < len(words)-1:

                ind_for_back_point = list(viterbi[j]).index(min(viterbi[j]))
                states.append(tags[int(back_point[j][ind_for_back_point])])

states = list(reversed(states))

for k in range(len(words)):
    print(words[k], ': ', states[k])


print(len(viterbi), len(words), len(tags))
# print(words[0], words[1])

# print(tags[list(viterbi[0]).index(min(viterbi[0]))], tags[list(viterbi[0]).index(min(viterbi[0]))])
