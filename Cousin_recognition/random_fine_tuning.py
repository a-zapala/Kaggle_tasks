import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import string
import random


train = pd.read_json('cooking_train.json')
test = pd.read_json('cooking_test.json')


def joining_preprocessor(line):
    return ' '.join(line).lower()


def get_word_freq(corpus, n_gram=(1, 1)):
    vec = CountVectorizer(preprocessor=joining_preprocessor, ngram_range=n_gram).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    return [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]


def get_top_n_words(corpus, n=None, n_gram=(1, 1)):
    words_freq = get_word_freq(corpus, n_gram)
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def get_not_used_again(train, test):
    wnl = WordNetLemmatizer()
    all_words_train = get_top_n_words(train['ingredients'])
    all_words_test = get_top_n_words(test['ingredients'])
    list_word_train = [wnl.lemmatize(t) for t, freq in all_words_train]
    list_word_test = [wnl.lemmatize(t) for t, freq in all_words_test]
    return list(set(list_word_train) - set(list_word_test))


not_used_again_word = get_not_used_again(train, test)


def preprocessor(line):
    without_not_used_ingredients = [word for word in line if not word in not_used_again_word]
    joining = ' '.join(without_not_used_ingredients).lower()
    without_punctuation = joining.translate(str.maketrans('', '', string.punctuation))
    return without_punctuation.translate(str.maketrans('', '', string.digits))


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.tkn = RegexpTokenizer(r'\w+')
        self.stopwords = stopwords.words('english')

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in self.tkn.tokenize(doc) if t not in self.stopwords]


vect = TfidfVectorizer(preprocessor=preprocessor, tokenizer=LemmaTokenizer())
X_train = vect.fit_transform(train['ingredients'])
y_train = train['cuisine']

param_grid = {
    'decision_function_shape': ['ovr', 'ovo'],
    'kernel': ['poly', 'rbf'],
    'degree': [3, 5],
    'shrinking': [True, False],
    'C': [0.001, 0.01, 0.1, 1, 10, 50, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'tol': [0.0001, 0.001, 0.01, 0.1]
}

number_iter = 200
i = 0
best_params = 0
best_scores = 0

for i in range(number_iter):
    params = {}
    for k in param_grid.keys():
        params[k] = random.choice(param_grid[k])

    res = cross_val_score(SVC(**params), X_train, y_train, n_jobs=-1, verbose=1)
    score = sum(res) / len(res)

    if score > best_scores:
        best_scores = score
        best_params = params
    print(i, score)
    print(params)
    print('BEST:', best_scores, best_params)
    with open("output/random_search.txt", "a") as myfile:
        myfile.write(str(i) + " " + str(score) + str(params) + " BEST" + str(best_scores) + str(
            best_params) + '\n')
print(best_params, best_scores)
