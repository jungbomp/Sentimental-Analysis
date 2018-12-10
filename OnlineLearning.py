import re
import random
import pickle
from math import exp, log
from datetime import datetime
from operator import itemgetter


def clean(s):
    """
        Returns a cleaned, lowercased string
    """
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def get_data(dataset, opts):
    """
    Running through data in an online manner
    Parses a file and yields label, identifier and features
    output:
            label: int, The label / target (set to "1" if test set)
            features: list of tuples, in the form [(hashed_feature_index,feature_value)]
    """
    f = open(dataset, "rb")
    for e, line in enumerate(f):
        label_map = {"negative": 0, "positive": 1}
        if e > 0:
            if not ((opts["set"] == "train" and (0 < e % 5)) or (opts["set"] == "test" and (e % 5 < 1))):
                continue

            r = line.decode('utf-8').strip().split('|')

            if opts["clean"]:
                r[1] = clean(r[1])

            # opts["D"] = 2 ** 25 = 33554432
            # Using hashing trick of Vowpal Wabbit.
            # It enables to use large scale feature space to store data applying fix length repression.
            features = [(hash(f) % opts["D"], 1) for f in r[1].split()]
            label = label_map[r[0]]

            # bygrams: Hashing ith feature and (i+1)th feature together.
            if opts["2grams"]:
                for i in range(len(features) - 1):
                    features.append((hash(str(features[i][0]) + str(features[i + 1][0])) % opts["D"], 1))
            yield label, features

    f.close()


def dot_product(features, weights):
    """
    Calculate dot product from features and weights
    input:
            features: A list of tuples [(feature_index,feature_value)]
            weights: the hashing trick weights filter,
            note: length is max(feature_index)
    output:
            dotp: the dot product
    """
    dotp = 0
    for f in features:
        dotp += weights[f[0]] * f[1]
    return dotp


def train_tron(dataset, opts):
    start = datetime.now()
    print("\nPass\t\tErrors\t\tAverage\t\tNr. Samples\tSince Start")

    # Initialize weight
    if opts["random_init"]:
        random.seed(3003)
        weights = [random.random()] * opts["D"]
    else:
        weights = [0.] * opts["D"]

    # Running training passes
    for pass_nr in range(opts["n_passes"]):
        error_counter = 0
        opts["set"] = "train"
        for e, (label, features) in enumerate(get_data(dataset, opts)):

            # 퍼셉트론은 지도학습 분류기의 일종이다.
            # 이전 값에 대한 학습으로 예측을 한다.
            # 내적(dotproduct) 값이 임계 값보다 높거나 낮은지에 따라
            # 초과하면 "1"을 예측하고 미만이면 "0"을 예측한다.
            dp = dot_product(features, weights) > 0.5

            # 다음 perceptron은 샘플의 레이블을 본다.
            # 실제 레이블 데이터에서 위 퍼셉트론으로 구한 dp값을 빼준다.
            # 예측이 정확하다면, error 값은 "0"이며, 가중치만 남겨 둔다.
            # 예측이 틀린 경우 error 값은 "1" 또는 "-1"이고 다음과 같이 가중치를 업데이트 한다.
            # weights[feature_index] += learning_rate * error * feature_value
            error = label - dp

            # 예측이 틀린 경우 퍼셉트론은 다음과 같이 가중치를 업데이트한다.
            if error != 0:
                error_counter += 1
                # Updating the weights
                for index, value in features:
                    weights[index] += opts["learning_rate"] * error * log(1. + value)

        # Reporting stuff
        print("%s\t\t%s\t\t%s\t\t%s\t\t%s" % ( \
            pass_nr + 1,
            error_counter,
            round(1 - error_counter / float(e + 1), 5),
            e + 1, datetime.now() - start))

        # Oh heh, we have overfit :)
        if error_counter == 0 or error_counter < opts["errors_satisfied"]:
            print("%s errors found during training, halting" % error_counter)
            break
    return weights


def test_tron(dataset, weights, opts):
    """
        output:
                preds: list, a list with
                [id,prediction,dotproduct,0-1normalized dotproduct]
    """
    start = datetime.now()
    print("\nTesting online\nErrors\t\tAverage\t\tNr. Samples\tSince Start")
    preds = []
    error_counter = 0
    opts["set"] = "test"
    for e, (label, features) in enumerate(get_data(dataset, opts)):

        dotp = dot_product(features, weights)
        # 내적이 0.5보다 크다면 긍정으로 예측한다.
        dp = dotp > 0.5
        if dp > 0.5:  # we predict positive class
            preds.append([e, 1, dotp])
        else:
            preds.append([e, 0, dotp])

        # get_data_tsv에서 테스트 데이터의 레이블을 1로 초기화 해주었음
        if label - dp != 0:
            error_counter += 1

    print("%s\t\t%s\t\t%s\t\t%s" % (
        error_counter,
        round(1 - error_counter / float(e + 1), 5),
        e + 1,
        datetime.now() - start))

    # normalizing dotproducts between 0 and 1
    # 내적을 구해 0과 1로 일반화 한다.
    # TODO: proper probability (bounded sigmoid?),
    # online normalization
    max_dotp = max(preds, key=itemgetter(2))[2]
    min_dotp = min(preds, key=itemgetter(2))[2]
    for p in preds:
        # appending normalized to predictions
        # 정규화 된 값을 마지막에 추가해 준다.
        # (피처와 가중치에 대한 내적값 - 최소 내적값) / 최대 내적값 - 최소 내적값
        # 이 값이 캐글에서 0.95의 AUC를 얻을 수 있는 값이다.
        p.append((p[2] - min_dotp) / float(max_dotp - min_dotp))

        # Reporting stuff
    print("Done testing in %s" % str(datetime.now() - start))

    # Write results to a file
    if True == opts["file_write"]:
        with open("data/OnlineLearning_{0:.5f}.csv".format(round(1 - error_counter / float(e + 1), 5)), "wb") as outfile:
            outfile.write('"id","sentiment"\n'.encode('utf-8'))
            for p in sorted(preds):
                outfile.write("{},{},{},{}\n".format(p[0], p[1],p[2],p[3]).encode('utf-8'))

    return preds


# Setting options
opts = {}
opts["D"] = 2 ** 25
opts["learning_rate"] = 0.1
opts["n_passes"] = 127 # Maximum number of passes to run before halting
opts["errors_satisfied"] = 0 # Halt when training errors < errors_satisfied
opts["random_init"] = False # set random weights, else set all 0
opts["clean"] = True # clean the text a little
opts["2grams"] = True # add 2grams
opts["file_write"] = True

# Load weight data from a pkl file
# infile = open("OnlineLearning_weight",'rb')
# weights = pickle.load(infile)
# infile.close()

#training and saving model into weights
weights = train_tron("data/reviews.csv", opts)

# Write weight data to a pkl file
# outfile = open("OnlineLearning_weight.pkl", 'wb')
# pickle.dump(weights, outfile)
# outfile.close()

# testing and saving predictions into preds
preds = test_tron("data/reviews.csv", weights, opts)
