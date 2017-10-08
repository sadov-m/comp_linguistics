# imports
# simple preprocessing + some info
import re
import os
from nltk import word_tokenize
from collections import Counter
"""import nltk
nltk.download_gui()
exit(0)"""
from nltk.corpus import stopwords
# more advanced preprocessing + training
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
# metrics and saving/loading a model
from sklearn.externals import joblib
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scl_opp_addition
# filters
reg_exp_url_finder = re.compile('(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
reg_exp_nickname_finder = re.compile('@[^\s|:]+')
reg_exp_amp_finder = re.compile('[^\"\'«]amp')
stop_words_list = set(stopwords.words('english'))
# paths
train_path = 'EI-reg-English-Train'
test_path = 'EI-reg-English-Dev'
# flags + emotion to process
current_emotion = 'sadness'
train_a_model = True  # False means evaluation only
no_feature_selection = True  # False means semi-automatic feature selection


def extracting_paths(path_dir):
    container = []

    for d, dirs, files in os.walk(path_dir):
        for f in files:
            filepath = os.path.join(d, f)  # формирование адреса
            container.append(filepath)  # добавление адреса в список

    return container


def get_texts_and_labels(path_to_txt):
    with open(path_to_txt, encoding='utf-8') as tweets_opener:
        data = [elem.split('\t') for elem in tweets_opener.read().split('\n')][:-1]
        texts = [tweet[1].lower() for tweet in data]
        labels = [float(tweet[-1]) for tweet in data]
        del data

    clean_texts = []
    for line in texts:
        text = reg_exp_url_finder.sub('', line)
        text = reg_exp_nickname_finder.sub('', text)
        text = reg_exp_amp_finder.sub('', text)
        # text = re.sub("[^a-z|\s]+", " ", text).split()
        text = re.sub("\s+", " ", text)
        # text = word_tokenize(text)
        clean_texts.append(text)

    return clean_texts, labels


def correct_labels(path_dir):
    X, y = [], []

    for file_path in extracting_paths(path_dir):
        temp_X, temp_Y = get_texts_and_labels(file_path)
        X += temp_X

        if file_path.find(current_emotion) != -1:
            y += temp_Y
        else:
            y += [0.0 for i in range(len(temp_Y))]

    return X, y


train_X, train_y = correct_labels(train_path)
test_X, test_y = correct_labels(test_path)

if no_feature_selection:
    vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1, 2), tokenizer=word_tokenize,
                                 stop_words=stop_words_list, min_df=2, max_df=0.8, max_features=50000)
else:
    hashtag_finder = re.compile('#\w+')
    extras_finder = re.compile('\D|\W+')
    extras = set(extras_finder.findall(' '.join(train_X)))-scl_opp_addition.all_punct_en
    features = hashtag_finder.findall(' '.join(train_X))
    new_features = []

    for key, value in Counter(features).most_common(1400):
        to_be_added = scl_opp_addition.segment(key[1:])

        if len(to_be_added) < 4:
            new_features.append(' '.join(to_be_added))

    new_features = set(new_features).union(extras)
    new_features.remove('')
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), vocabulary=new_features, max_features=50000)

train_X = vectorizer.fit_transform(train_X)
test_X = vectorizer.transform(test_X)
print('number of features used is:', len(vectorizer.get_feature_names()))

if __name__ == '__main__':
    # learning
    if train_a_model:
        model = Ridge()
        model.fit(train_X, train_y)
        # joblib.dump(model, 'ridge_model_feature_selection')
    else:
        path_to_a_model = input('type in a path to a model:')
        model = joblib.load(path_to_a_model)

    pred_y = model.predict(test_X)
    score = model.score(test_X, test_y)

    print('rho-value is:', spearmanr(test_y, pred_y))
    print('rmse is: ', mean_squared_error(test_y, pred_y) ** 0.5)
    print('the score is:', score)

    fig, ax = plt.subplots()
    ax.scatter(test_y, pred_y)
    minimum, maximum = min(test_y), max(test_y)
    ax.plot([minimum, maximum], [minimum, maximum], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
