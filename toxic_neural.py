#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import itertools
import re

from utils.treebank import StanfordSentiment
import utils.glove as glove
import csv
from random import shuffle

from q3_sgd import load_saved_params, sgd

# We will use sklearn here because it will run faster than implementing
# ourselves. However, for other parts of this assignment you must implement
# the functions yourself!
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


def getArguments():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pretrained", dest="pretrained", action="store_true",
                       help="Use pretrained GloVe vectors.")
    group.add_argument("--yourvectors", dest="yourvectors", action="store_true",
                       help="Use your vectors from q3.")
    return parser.parse_args()


def getSentenceFeatures(tokens, wordVectors, sentence):
    """
    Obtain the sentence feature for sentiment analysis by averaging its
    word vectors
    """

    # Implement computation for the sentence features given a sentence.

    # Inputs:
    # tokens -- a dictionary that maps words to their indices in
    #           the word vector list
    # wordVectors -- word vectors (each row) for all tokens
    # sentence -- a list of words in the sentence of interest

    # Output:
    # - sentVector: feature vector for the sentence

    sentVector = np.zeros((wordVectors.shape[1],))
    count = 0.0

    ### YOUR CODE HERE
    for word in sentence:
        if word in tokens:
            sentVector += wordVectors[tokens[word]]
            count += 1.0

    sentVector = np.divide(sentVector, count)
    #raise NotImplementedError
    ### END YOUR CODE

    assert sentVector.shape == (wordVectors.shape[1],)
    return sentVector


def getRegularizationValues():
    """Try different regularizations

    Return a sorted list of values to try.
    """
    values = None   # Assign a list of floats in the block below
    ### YOUR CODE HERE
    
    values = [2, 1, 0.0, 0.1, 0.01, 0.001, 0.0005, 0.0001, 0.00005]
    ### END YOUR CODE
    return sorted(values)


def chooseBestModel(results):
    """Choose the best model based on dev set performance.

    Arguments:
    results -- A list of python dictionaries of the following format:
        {
            "reg": regularization,
            "clf": classifier,
            "train": trainAccuracy,
            "dev": devAccuracy,
            "test": testAccuracy
        }

    Each dictionary represents the performance of one model.

    Returns:
    Your chosen result dictionary.
    """
    bestResult = None

    ### YOUR CODE HERE
    bestAccuracy = 0.0
    for result in results:
        # Get best test accuracy that isn't overfitting to test data.
        if result["test"] > bestAccuracy and result["test"] < result["train"]:
            bestResult = result
            bestAccuracy = result['test']
    ### END YOUR CODE

    return bestResult


def accuracy(y, yhat):
    """ Precision for classifier """
    assert(y.shape == yhat.shape)
    return np.sum(y == yhat) * 100.0 / y.size


def plotRegVsAccuracy(regValues, results, filename):
    """ Make a plot of regularization vs accuracy """
    plt.plot(regValues, [x["train"] for x in results])
    plt.plot(regValues, [x["dev"] for x in results])
    plt.xscale('log')
    plt.xlabel("regularization")
    plt.ylabel("accuracy")
    plt.legend(['train', 'dev'], loc='upper left')
    plt.savefig(filename)


def outputConfusionMatrix(features, labels, clf, filename):
    """ Generate a confusion matrix """
    pred = clf.predict(features)
    cm = confusion_matrix(labels, pred, labels=range(5))
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.colorbar()
    classes = ["- -", "-", "neut", "+", "+ +"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)


def outputPredictions(dataset, features, labels, clf, filename):
    """ Write the predictions to file """
    pred = clf.predict(features)
    with open(filename, "w") as f:
        print >> f, "True\tPredicted\tText"
        for i in xrange(len(dataset)):
            print >> f, "%d\t%d\t%s" % (
                labels[i], pred[i], " ".join(dataset[i][0]))

def getToxicDataMultilabel():
    data = []
    with open('multilabel.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        count = 0
        for i, row in enumerate(reader):

            if count == 0:
                count += 1
                continue

            val = ([x.lower() for x in re.findall(r"[\w']+|[.,!?;]", row[1])], [])

            if int(row[2]) != 0:
                val[1].append(0)

            if int(row[3]) != 0:
                val[1].append(1)

            if int(row[4]) != 0:
                val[1].append(2)

            if int(row[5]) != 0:
                val[1].append(3)

            if int(row[6]) != 0:
                val[1].append(4)

            if int(row[7]) != 0:
                val[1].append(5)

            if len(val[1]) == 0:
                val[1].append(6)

            data.append(val)

    tokens = dict()
    tokenfreq = dict()
    wordcount = 0
    revtokens = []
    idx = 0

    for row in data:
        sentence = row[0]
        for w in sentence:
            wordcount += 1
            if not w in tokens:
                tokens[w] = idx
                revtokens += [w]
                tokenfreq[w] = 1
                idx += 1
            else:
                tokenfreq[w] += 1

    tokens["UNK"] = idx
    revtokens += ["UNK"]
    tokenfreq["UNK"] = 1
    wordcount += 1

    return data, tokens, 6


def main(args):
    """ Train a model to do sentiment analyis"""
    dataset, tokens, num_labels = getToxicDataMultilabel()
    target_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'non_toxic']

    # Shuffle data
    shuffle(dataset)

    num_data = len(dataset)

    # Create train, dev, and test
    train_cutoff = int(0.6 * num_data)
    dev_start = int(0.6 * num_data) + 1
    dev_cutoff = int(0.8 * num_data)

    trainset = dataset[:train_cutoff]
    devset = dataset[dev_start:dev_cutoff]
    testset = dataset[dev_cutoff + 1:]

    nWords = len(tokens)
    wordVectors = glove.loadWordVectors(tokens)
    dimVectors = wordVectors.shape[1]

    # Load the train set
    #trainset = dataset.getTrainSentences()
    nTrain = len(trainset)
    trainFeatures = np.zeros((nTrain, dimVectors))
    trainLabels = []


    for i in xrange(nTrain):
        words = trainset[i][0]
        trainLabels.append(trainset[i][1])
        trainFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

    # Prepare dev set features
    nDev = len(devset)
    devFeatures = np.zeros((nDev, dimVectors))
    devLabels = []
    for i in xrange(nDev):
        words = devset[i][0]
        devLabels.append(devset[i][1])
        devFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

    # Prepare test set features
    #testset = dataset.getTestSentences()
    nTest = len(testset)
    testFeatures = np.zeros((nTest, dimVectors))
    testLabels = []
    for i in xrange(nTest):
        words = testset[i][0]
        testLabels.append(testset[i][1])
        testFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)


    # We will save our results from each run
    results = []
    regValues = getRegularizationValues()
    print "LR Results:"
    classifier = Pipeline([
        ('vectorizer', CountVectorizer(min_n=1,max_n=2)),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LinearSVC()))])

    classifier.fit(trainFeatures, trainLabels)
    predicted = classifier.predict(devFeatures)
    clf = SVC()
    clf.fit(trainFeatures, trainLabels)

    # Test on train set
    pred = clf.predict(trainFeatures)
    trainAccuracy = accuracy(trainLabels, pred)
    print "Train accuracy (%%): %f" % trainAccuracy

    # Test on dev set
    pred = clf.predict(devFeatures)
    devAccuracy = accuracy(devLabels, pred)
    print "Dev accuracy (%%): %f" % devAccuracy

    # Test on test set
    # Note: always running on test is poor style. Typically, you should
    # do this only after validation.
    pred = clf.predict(testFeatures)
    testAccuracy = accuracy(testLabels, pred)
    print "Test accuracy (%%): %f" % testAccuracy

    results.append({
        "reg": 0.0,
        "clf": clf,
        "train": trainAccuracy,
        "dev": devAccuracy,
        "test": testAccuracy})

    # Print the accuracies
    print ""
    print "=== Recap ==="
    print "Reg\t\tTrain\tDev\tTest"
    for result in results:
        print "%.2E\t%.3f\t%.3f\t%.3f" % (
            result["reg"],
            result["train"],
            result["dev"],
            result["test"])
    print ""

    bestResult = chooseBestModel(results)
    # print "Best regularization value: %0.2E" % bestResult["reg"]
    # print "Test accuracy (%%): %f" % bestResult["test"]

    # do some error analysis
    if args.pretrained:
        plotRegVsAccuracy(regValues, results, "q4_reg_v_acc.png")
        outputConfusionMatrix(devFeatures, devLabels, bestResult["clf"],
                              "q4_dev_svm_conf.png")
        outputPredictions(devset, devFeatures, devLabels, bestResult["clf"],
                          "q4_dev_svm_pred.txt")


if __name__ == "__main__":
    main(getArguments())
