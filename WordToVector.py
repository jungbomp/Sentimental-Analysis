import re
import pandas as pd
import nltk
import nltk.data
import numpy as np
import logging

# import html5lib
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from gensim.models import word2vec
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

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

    # 7. Return a list of words
    return words


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
    # nltk.download('punkt')
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


def makeFeatureVec(words, model, num_features):
    """
        Compute average of word vector in the sentence
    """
    # Initialize
    featureVec = np.zeros((num_features,),dtype="float32")

    nwords = 0.
    # Index2word is a list which contains words in the model dictionary
    # Initialize
    index2word_set = set(model.wv.index2word)
    # Add words contained the model dictionary
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # Compute average
    if 0 < nwords:
        featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    """
        Compute average feature of each word list and return 2D numpy array
    :param reviews:
    :param model:
    :param num_features:
    :return:
    """

    # Initialize counter
    counter = 0.
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:
        # print every 1000
        if counter % 1000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))
        reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model,num_features)

        counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    # clean_reviews = train['text'].apply(review_to_words)
    clean_reviews = apply_by_multiprocessing(reviews["text"], review_to_words, workers=4)
    return clean_reviews


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load text data
print('Load review data...')
start = datetime.now()
reviews = pd.read_csv('data/reviews_4K.csv', header=0, delimiter='|', quoting=3)
# Extract train data for every four rows and test data for every fifth row
train = reviews[0 < (reviews.index + 1) % 5]
train_sentiment = label_to_sentiment(train['label'])
test = reviews[0 == (reviews.index + 1) % 5]
test_sentiment = label_to_sentiment(test['label'])
print('End to load review data after {0}'.format(datetime.now() - start))

# Extract all words
print('Extract all words...')
start = datetime.now()
sentences = []
for text in train["text"]:
    sentences += review_to_sentences(text, remove_stopwords=False)
print('End a job after {0}'.format(datetime.now() - start))

# Words to vector (model learning)
print('Words to vector (model learning)...')
start = datetime.now()
num_features = 300 
model = word2vec.Word2Vec(sentences,
                          workers=4,
                          size=num_features,
                          min_count=40,
                          window=10,
                          sample=1e-3
                          )
# Release memory after vectoring
model.init_sims(replace=True)
print('End a job after {0}'.format(datetime.now() - start))

# Get average feature vectors of train data
print("Get average feature vectors of train data...")
start = datetime.now()
trainDataVecs = getAvgFeatureVecs(getCleanReviews(train), model, num_features)
print('End a job after {0}'.format(datetime.now() - start))

# Get average feature vectors of train data
print("Get average feature vectors of test data...")
start = datetime.now()
testDataVecs = getAvgFeatureVecs(getCleanReviews(test), model, num_features)
print('End a job after {0}'.format(datetime.now() - start))

print("Learning...")
start = datetime.now()
# clf = DecisionTreeClassifier(criterion='gini', random_state=1, max_depth=3, min_samples_leaf=5)
clf = DecisionTreeClassifier()
clf = clf.fit(trainDataVecs, train_sentiment)
print('End learning after {0}'.format(datetime.now() - start))

# Predicting test vector
print('Predicting test vector...')
start = datetime.now()
Y_predict = clf.predict(testDataVecs)
print('End Predicting after {0}'.format(datetime.now() - start))
print(accuracy_score(test_sentiment, Y_predict))
output = pd.DataFrame(data={'sentiment': Y_predict}, columns=['sentiment'])

print('Write result data...')
start = datetime.now()
output.to_csv('data/DecisionTreeWithW2V_{0:.5f}.csv'.format(accuracy_score(test_sentiment, Y_predict)), index=True, quoting=3)

# Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale data.
# Note that it must apply the same scaling to the test set for meaningful results. There are a lot of
# different methods for normalization of data, here will use the built-in StandardScaler for standardization.
# Normalization
print('Normalizing data')
start = datetime.now()
scaler = StandardScaler()
scaler.fit(trainDataVecs)
X_train = scaler.transform(trainDataVecs)
X_test = scaler.transform(testDataVecs)
print('End normalizing after {0}'.format(datetime.now() - start))

print("Learning...")
start = datetime.now()
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
# clf = MLPClassifier()
start = datetime.now()
clf = clf.fit(X_train, train_sentiment)
print('End learning after {0}'.format(datetime.now() - start))

# score = np.mean(cross_val_score(clf, trainDataVecs, train_sentiment, cv=10, scoring='roc_auc'))

# Predicting
print('Predicting...')
start = datetime.now()
Y_predict = clf.predict(X_test)
print('End predicting after {0}'.format(datetime.now() - start))
print(accuracy_score(test_sentiment, Y_predict))

# Write result to a csv file.
print('Write result data...')
start = datetime.now()
output = pd.DataFrame(data={'sentiment': Y_predict}, columns=['sentiment'])
output.to_csv('data/MLPwithW2V_{0:.5f}.csv'.format(accuracy_score(test_sentiment, Y_predict)), index=True, quoting=3)
print('End writing after {0}'.format(datetime.now() - start))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(test_sentiment, Y_predict))
print(classification_report(test_sentiment, Y_predict))
