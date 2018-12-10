import re
import pandas as pd
import nltk
import nltk.data
import numpy as np

# import html5lib
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.externals import joblib

from multiprocessing import Pool
from datetime import datetime


def label_to_sentiment(labels):
    """

    :param labels:
    :return:
    """

    num = {'negative': 0.00, 'positive': 1.00}
    sentiment = []
    for l in labels:
        sentiment.append(num[l])

    return sentiment


def review_to_words(raw_review, remove_stopwords=False):
    """
        Function to convert a document to a sequence of words,
        optionally removing stop words. Returns a list of words.
    """

    # 1. Remove HTML -- It should import html5lib in order to download html.parser
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()

    # 2. Remove non-letters (Replace non-letters to white space)
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)

    # 3. Convert words to lower case and split them
    words = letters_only.lower().split()

    # 4. Optionally remove stop words (false by default, It is faster to searching on set than list in Python)
    if remove_stopwords:
        nltk.download('stopwords')
        stops = set(stopwords.words('english'))
        # 5. Remove stop words
        words = [w for w in words if w not in stops]

    # 6. Stmmer
    stemmer = SnowballStemmer('english')
    words = [stemmer.stem(w) for w in words]

    # 7. Return a string separating words with a white space
    return ' '.join(words)


# Define a function to split a review into parsed sentences
def review_to_sentences(review, remove_stopwords=False):
    """
        Function to split a review into parsed sentences. Returns
        a list of sentences, where each sentences is a list of words
    :param review:
    :param remove_stopwords:
    :return:
    """

    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    # Load the punkt tokenizer
    nltk.download('punkt')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(review.strip())

    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_words to get a list of words
            sentences.append(review_to_words(raw_sentence, remove_stopwords))

    # Return the list of sentences (each sentence is a list of words, so this return a list of lists
    return sentences


# Parallel processing
def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, **kwargs):
    # Acquare workers parameter
    workers = kwargs.pop('workers')
    # Define pool with workers count
    pool = Pool(processes=workers)
    # Divide a function and data frame by workers count
    result = pool.map(_apply_df, [(d, func, kwargs) for d in np.array_split(df, workers)])
    pool.close()
    # Combine the result and return
    return pd.concat(list(result))


# Load text data
print('Load review data...')
start = datetime.now()
reviews = pd.read_csv('data/reviews.csv', header=0, delimiter='|', quoting=3)
# Extract train data for every four rows and test data for every fifth row
train = reviews[0 < (reviews.index + 1) % 5]
train_sentiment = label_to_sentiment(train['label'])
test = reviews[0 == (reviews.index + 1) % 5]
test_sentiment = label_to_sentiment(test['label'])
print('End to load review data after {0}'.format(datetime.now() - start))

# Tokenize
print('Tokenize reviews to words...')
start = datetime.now()
train_data_features = apply_by_multiprocessing(train['text'], review_to_words, workers=4)
# clean_train_reviews = train['text'].apply(review_to_words)
test_data_features = apply_by_multiprocessing(test['text'], review_to_words, workers=4)
# clean_test_reviews = test['text'].apply(review_to_words)
print('End tokenizing after {0}'.format(datetime.now() - start))

vectorizer = CountVectorizer(analyzer='word',
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             min_df=2, # Minimum number of documents containing token
                             ngram_range=(1, 3),
                             max_features=20000
                            )

print('Vectoring train words...')
start = datetime.now()
train_data_features = vectorizer.fit_transform(train_data_features)
print('End vectoring after {0}'.format(datetime.now() - start))
joblib.dump(train_data_features, 'BagsOfWords_train_vec.pkl')
joblib.dump(train_sentiment, 'BagsOfWords_train_sentiment.pkl')

#print('Downscaling(Term Frequency times Inverse Document Frequency)...')
#start = datetime.now()
# tfidf_transformer = TfidfTransformer()
# train_data_features = tfidf_transformer.fit_transform(train_data_features)
#print('End downscaling after {0}'.format(datetime.now() - start))

# Use pipeline in order to improve running performance
# clf = Pipeline([('vect', vectorizer),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', clf)
#                      ])
# clf = clf.fit(train_data_features, train_sentiment)

# Learning using MLP
print("Learning using MLP...")
start = datetime.now()
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf = MLPClassifier()
clf = clf.fit(train_data_features, train_sentiment)
print('End learning after {0}'.format(datetime.now() - start))
joblib.dump(clf, 'MLPwithBagsOfWords_trained_clf.pkl')

# Vectoring test words
print('Vectoring test words...')
start = datetime.now()
test_data_features = vectorizer.transform(test_data_features)
test_data_features = test_data_features.toarray()
print('End vectoring after {0}'.format(datetime.now() - start))
joblib.dump(test_data_features, 'BagsOfWords_test_vec.pkl')
joblib.dump(test_sentiment, 'BagsOfWords_test_sentiment.pkl')

# Predicting test vector
print('Predicting test vector...')
start = datetime.now()
Y_predict = clf.predict(test_data_features)
print('End Predicting after {0}'.format(datetime.now() - start))
print(accuracy_score(test_sentiment, Y_predict))

print('Write result data...')
start = datetime.now()
output = pd.DataFrame(data={'sentiment': Y_predict}, columns=["sentiment"])
output.to_csv('data/MLPwithBOW_{0:.5f}.csv'.format(accuracy_score(test_sentiment, Y_predict)), index=True, quoting=3)
print('End writing after {0}'.format(datetime.now() - start))

# Learning
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=2018)
print('Learning words...')
start = datetime.now()
clf.fit(train_data_features, train_sentiment)
print('End learning after {0}'.format(datetime.now() - start))
joblib.dump(clf, 'RandomForestwithBagsOfWords_trained_clf.pkl')

# score = np.mean(cross_val_score(clf, train_data_features, train_sentiment, cv=10, scoring='roc_auc'))

# Predicting
print('Predicting test vector...')
start = datetime.now()
result = clf.predict(test_data_features)
print('End vectoring after {0}'.format(datetime.now() - start))
print(accuracy_score(test_sentiment, result))

# Save result into csv file
print('Write result data...')
start = datetime.now()
output = pd.DataFrame(data={'id': test.index, 'sentiment': result}, columns=["sentiment"])
output.head()
output.to_csv('data/RandomForestWithBOW_{0:.5f}.csv'.format(accuracy_score(test_sentiment, result)), index=False, quoting=3)
print('End writing after {0}'.format(datetime.now() - start))

# clf = DecisionTreeClassifier(criterion='gini', random_state=1, max_depth=3, min_samples_leaf=5)
clf = DecisionTreeClassifier()
print("Learning...")
start = datetime.now()
clf = clf.fit(train_data_features, train_sentiment)
joblib.dump(clf, 'DecisionTreewithBagsOfWords_trained_clf.pkl')
Y_predict = clf.predict(test_data_features)
print(accuracy_score(test_sentiment, Y_predict))
output = pd.DataFrame(data={'sentiment': Y_predict}, columns=['sentiment'])
# output.head()
output.to_csv('data/DecisionTreeWithBOW_{0:.5f}.csv'.format(accuracy_score(test_sentiment, Y_predict)), index=True, quoting=3)
