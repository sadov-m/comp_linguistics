import sklearn_crfsuite
from sklearn_crfsuite.metrics import *
from collections import Counter
from sklearn.externals import joblib
from nltk.corpus import stopwords

show_top_features = True
train_a_clf = True

stop_words = set(stopwords.words('english'))


def open_file(filename):
    with open(filename, encoding='utf-8') as data_opener:
        data = data_opener.read().split('\n')[2:]
    return data


# columns: word/pos_tag/chunk/ner_tag
train_data = open_file('CoNLL-2003/eng.train')
test_data = open_file('CoNLL-2003/eng.testa') + open_file('CoNLL-2003/eng.testb')


def get_sents(data):
    sents = []
    tmp_sent = []

    for line in data:

        if line != '':
            tmp_sent.append(line.split(' '))
        else:
            sents.append(tmp_sent)
            tmp_sent = []

    return sents


train_sents = get_sents(train_data)
test_sents = get_sents(test_data)


def word2features_baseline(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
        }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
    else:
        features['EOS'] = True

    return features


def word2features_my_version(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    # getting word_mask/shape
    def get_word_mask(target_word):
        word_mask = ''

        for symbol in target_word:

            if symbol.isdigit():
                word_mask += 'd'
            elif symbol.isupper():
                word_mask += 'X'
            elif symbol.islower():
                word_mask += 'x'
            else:
                word_mask += 'U'

        return word_mask

    features = {
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
            # new features:
            'word_mask': get_word_mask(word),
            'word_mask_short': get_word_mask(word)[:2],
            'has_cyphen': '-' in word,
            'has_dot': '.' in word,
            'is_aux': word in stop_words
        }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
                # new features:
                '-1:word_mask': get_word_mask(word1),
                '-1:word_mask_short': get_word_mask(word1)[:2],
                '-1:has_cyphen': '-' in word1,
                '-1:has_dot': '.' in word1,
                '-1:is_aux': word1 in stop_words
            })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
                # new features:
                '+1:word_mask': get_word_mask(word1),
                '+1:word_mask_short': get_word_mask(word1)[:2],
                '+1:has_cyphen': '-' in word1,
                '+1:has_dot': '.' in word1,
                '+1:is_aux': word1 in stop_words
            })
    else:
        features['EOS'] = True

    return features


def sent2features(sent, feat_extraction_function):
    return [feat_extraction_function(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, chunk, label in sent]


def sent2tokens(sent):
    return [token for token, postag, chunk, label in sent]


X_train = [sent2features(s, word2features_my_version) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s, word2features_my_version) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

# print(set(' '.join([' '.join(y_sample) for y_sample in y_test]).split(' ')))

if train_a_clf:
    crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True)
    crf.fit(X_train, y_train)
    model_name = input('please enter name for saving model:')
    joblib.dump(crf, model_name)
else:
    model_name = input('please enter the name of a loaded model:')
    crf = joblib.load(model_name)

labels = list(crf.classes_)
labels.remove('O')

y_pred = crf.predict(X_test)
flat_f1_score(y_test, y_pred, average='weighted', labels=labels)

# group B and I results
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0]))

print(flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))


def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))


if show_top_features:
    print("Top positive:")
    print_state_features(Counter(crf.state_features_).most_common(30))

    print("\nTop negative:")
    print_state_features(Counter(crf.state_features_).most_common()[-30:])
