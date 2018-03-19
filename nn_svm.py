# #!/usr/bin/env python

# import argparse
# import numpy as np
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
# import itertools
# import re

# from utils.treebank import StanfordSentiment
# import utils.glove as glove
# import csv
# from random import shuffle

# from q3_sgd import load_saved_params, sgd

# # We will use sklearn here because it will run faster than implementing
# # ourselves. However, for other parts of this assignment you must implement
# # the functions yourself!
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix
# from sklearn.svm import SVC


# def getArguments():
#     parser = argparse.ArgumentParser()
#     group = parser.add_mutually_exclusive_group(required=True)
#     group.add_argument("--pretrained", dest="pretrained", action="store_true",
#                        help="Use pretrained GloVe vectors.")
#     group.add_argument("--yourvectors", dest="yourvectors", action="store_true",
#                        help="Use your vectors from q3.")
#     return parser.parse_args()


# def getSentenceFeatures(tokens, wordVectors, sentence):
#     """
#     Obtain the sentence feature for sentiment analysis by averaging its
#     word vectors
#     """

#     # Implement computation for the sentence features given a sentence.

#     # Inputs:
#     # tokens -- a dictionary that maps words to their indices in
#     #           the word vector list
#     # wordVectors -- word vectors (each row) for all tokens
#     # sentence -- a list of words in the sentence of interest

#     # Output:
#     # - sentVector: feature vector for the sentence

#     sentVector = np.zeros((wordVectors.shape[1],))
#     count = 0.0
#     ### YOUR CODE HERE
#     for word in sentence:
#         if word in tokens:
#             sentVector += wordVectors[tokens[word]]
#             count += 1.0
#     sentVector = np.divide(sentVector, count)
    
#     #raise NotImplementedError
#     ### END YOUR CODE

#     assert sentVector.shape == (wordVectors.shape[1],)
#     return sentVector


# def getLSTMSentenceFeatures(tokens, wordVectors, dimVectors, sentence, maxLength):
#     """
#     Obtain the sentence feature for sentiment analysis by averaging its
#     word vectors
#     """

#     # Implement computation for the sentence features given a sentence.

#     # Inputs:
#     # tokens -- a dictionary that maps words to their indices in
#     #           the word vector list
#     # wordVectors -- word vectors (each row) for all tokens
#     # sentence -- a list of words in the sentence of interest

#     # Output:
#     # - sentVector: feature vector for the sentence\\


#     sentVector = np.zeros((maxLength, dimVectors))

#     ### YOUR CODE HERE
#     for i in range(0, len(sentence)):
#         word = sentence[i]
#         if i == maxLength:
#             break
#         if word in tokens:
#             sentVector[i] = wordVectors[tokens[word]]

#     return sentVector


# def getMLPSentenceFeatures(tokens, wordVectors, dimVectors, sentence, maxLength):
#     """
#     Obtain the sentence feature for sentiment analysis by averaging its
#     word vectors
#     """

#     # Implement computation for the sentence features given a sentence.

#     # Inputs:
#     # tokens -- a dictionary that maps words to their indices in
#     #           the word vector list
#     # wordVectors -- word vectors (each row) for all tokens
#     # sentence -- a list of words in the sentence of interest

#     # Output:
#     # - sentVector: feature vector for the sentence\\


#     sentVector = np.zeros((dimVectors,))
#     count = 0.0
#     ### YOUR CODE HERE
#     for i in range(0, len(sentence)):
#         word = sentence[i]
#         if word in tokens:
#             sentVector += wordVectors[tokens[word]]
#             count += 1.0
#     sentVector = np.divide(sentVector, count)
#     return sentVector


# def getRegularizationValues():
#     """Try different regularizations

#     Return a sorted list of values to try.
#     """
#     values = None   # Assign a list of floats in the block below
#     ### YOUR CODE HERE
    
#     values = [2, 1, 0.0, 0.1, 0.01, 0.001, 0.0005, 0.0001, 0.00005]
#     ### END YOUR CODE
#     return sorted(values)


# def chooseBestModel(results):
#     """Choose the best model based on dev set performance.

#     Arguments:
#     results -- A list of python dictionaries of the following format:
#         {
#             "reg": regularization,
#             "clf": classifier,
#             "train": trainAccuracy,
#             "dev": devAccuracy,
#             "test": testAccuracy
#         }

#     Each dictionary represents the performance of one model.

#     Returns:
#     Your chosen result dictionary.
#     """
#     bestResult = None

#     ### YOUR CODE HERE
#     bestAccuracy = 0.0
#     for result in results:
#         # Get best test accuracy that isn't overfitting to test data.
#         if result["test"] > bestAccuracy and result["test"] < result["train"]:
#             bestResult = result
#             bestAccuracy = result['test']
#     ### END YOUR CODE

#     return bestResult

# def accuracy(y, yhat):
#     """ Precision for classifier """
#     assert(y.shape == yhat.shape)
#     return np.sum(y == yhat) * 100.0 / y.size

# def plotRegVsAccuracy(regValues, results, filename):
#     """ Make a plot of regularization vs accuracy """
#     plt.plot(regValues, [x["train"] for x in results])
#     plt.plot(regValues, [x["dev"] for x in results])
#     plt.xscale('log')
#     plt.xlabel("regularization")
#     plt.ylabel("accuracy")
#     plt.legend(['train', 'dev'], loc='upper left')
#     plt.savefig(filename)


# def outputConfusionMatrix(features, labels, clf, filename):
#     """ Generate a confusion matrix """
#     pred = clf.predict(features)
#     cm = confusion_matrix(labels, pred, labels=range(5))
#     plt.figure()
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
#     plt.colorbar()
#     classes = ["- -", "-", "neut", "+", "+ +"]
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes)
#     plt.yticks(tick_marks, classes)
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.savefig(filename)


# def outputPredictions(dataset, features, labels, clf, filename):
#     """ Write the predictions to file """
#     pred = clf.predict(features)
#     with open(filename, "w") as f:
#         print >> f, "True\tPredicted\tText"
#         for i in xrange(len(dataset)):
#             print >> f, "%d\t%d\t%s" % (
#                 labels[i], pred[i], " ".join(dataset[i][0]))

# def getToxicData():
#     data = []
#     maxSentence = 0
#     with open('balanced_data_kevin.csv', 'rb') as csvfile:
#         reader = csv.reader(csvfile, delimiter=',')
#         count = 0
#         for i, row in enumerate(reader):
#             if count == 0:
#                 count += 1
#                 continue
#             data.append(([x.lower() for x in re.findall(r"[\w']+|[.,!?;]", row[1])], int(row[2])))

#     tokens = dict()
#     tokenfreq = dict()
#     wordcount = 0
#     revtokens = []
#     idx = 1
#     for row in data:
#         sentence = row[0]
#         if len(sentence) > maxSentence:
#             maxSentence = len(sentence)

#         for w in sentence:
#             wordcount += 1
#             if not w in tokens:
#                 tokens[w] = idx
#                 revtokens += [w]
#                 tokenfreq[w] = 1
#                 idx += 1
#             else:
#                 tokenfreq[w] += 1

#     tokens["    "] = 0
#     revtokens += ["UNK"]
#     tokenfreq["UNK"] = 1
#     wordcount += 1

#     return data, tokens, maxSentence

# def getLSTMData():
#     dataset, tokens, maxLength= getToxicData()
#     # Shuffle data
#     shuffle(dataset)

#     maxLength = 100

#     num_data = len(dataset)

#     # Create train, dev, and test
#     train_cutoff = int(0.6 * num_data)
#     dev_start = int(0.6 * num_data) + 1
#     dev_cutoff = int(0.8 * num_data)

#     trainset = dataset[:train_cutoff]
#     devset = dataset[dev_start:dev_cutoff]
#     testset = dataset[dev_cutoff + 1:]

#     nWords = len(tokens)

#     wordVectors = glove.loadWordVectors(tokens)
#     dimVectors = wordVectors.shape[1]

#     # Load the train set
#     #trainset = dataset.getTrainSentences()
#     nTrain = len(trainset)
#     trainFeatures = np.zeros((nTrain, maxLength, dimVectors))
#     trainLabels = np.zeros((nTrain,), dtype=np.int32)


#     for i in xrange(nTrain):
#         words, trainLabels[i] = trainset[i]
#         trainFeatures[i, :] = getLSTMSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

#     # Prepare dev set features
#     #devset = dataset.getDevSentences()
#     nDev = len(devset)
#     devFeatures = np.zeros((nDev, maxLength, dimVectors))
#     devLabels = np.zeros((nDev,), dtype=np.int32)
#     for i in xrange(nDev):
#         words, devLabels[i] = devset[i]
#         devFeatures[i, :] = getLSTMSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

#     # Prepare test set features
#     #testset = dataset.getTestSentences()
#     nTest = len(testset)
#     testFeatures = np.zeros((nTest, maxLength, dimVectors))
#     testLabels = np.zeros((nTest,), dtype=np.int32)
#     for i in xrange(nTest):
#         words, testLabels[i] = testset[i]
#         testFeatures[i, :] = getLSTMSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

#     return trainFeatures, trainLabels, devFeatures, devLabels, testFeatures, testLabels, maxLength, dimVectors


# # def getLSTMData():
# #     dataset, tokens, maxLength= getToxicData()
# #     # Shuffle data
# #     shuffle(dataset)

# #     maxLength = 500

# #     num_data = len(dataset)

# #     # Create train, dev, and test
# #     train_cutoff = int(0.6 * num_data)
# #     dev_start = int(0.6 * num_data) + 1
# #     dev_cutoff = int(0.8 * num_data)

# #     trainset = dataset[:train_cutoff]
# #     devset = dataset[dev_start:dev_cutoff]
# #     testset = dataset[dev_cutoff + 1:]

# #     nWords = len(tokens)

# #     wordVectors = glove.loadWordVectors(tokens)
# #     dimVectors = wordVectors.shape[1]

# #     # Load the train set
# #     #trainset = dataset.getTrainSentences()
# #     nTrain = len(trainset)
# #     trainFeatures = np.zeros((nTrain, maxLength, dimVectors))
# #     trainLabels = np.zeros((nTrain,), dtype=np.int32)


# #     for i in xrange(nTrain):
# #         words, trainLabels[i] = trainset[i]
# #         trainFeatures[i, :] = getLSTMSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

# #     # Prepare dev set features
# #     #devset = dataset.getDevSentences()
# #     nDev = len(devset)
# #     devFeatures = np.zeros((nDev, maxLength, dimVectors))
# #     devLabels = np.zeros((nDev,), dtype=np.int32)
# #     for i in xrange(nDev):
# #         words, devLabels[i] = devset[i]
# #         devFeatures[i, :] = getLSTMSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

# #     # Prepare test set features
# #     #testset = dataset.getTestSentences()
# #     nTest = len(testset)
# #     testFeatures = np.zeros((nTest, maxLength, dimVectors))
# #     testLabels = np.zeros((nTest,), dtype=np.int32)
# #     for i in xrange(nTest):
# #         words, testLabels[i] = testset[i]
# #         testFeatures[i, :] = getLSTMSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

# #     return trainFeatures, trainLabels, devFeatures, devLabels, testFeatures, testLabels, maxLength, dimVectors




#-----------------------------------------------------------






# def getMLPData():
#     dataset, tokens, maxLength= getToxicData()
#     # Shuffle data
#     shuffle(dataset)

#     maxLength = 100

#     num_data = len(dataset)

#     # Create train, dev, and test
#     train_cutoff = int(0.6 * num_data)
#     dev_start = int(0.6 * num_data) + 1
#     dev_cutoff = int(0.8 * num_data)

#     trainset = dataset[:train_cutoff]
#     devset = dataset[dev_start:dev_cutoff]
#     testset = dataset[dev_cutoff + 1:]

#     nWords = len(tokens)

#     wordVectors = glove.loadWordVectors(tokens)
#     dimVectors = wordVectors.shape[1]

#     # Load the train set
#     #trainset = dataset.getTrainSentences()
#     nTrain = len(trainset)
#     trainFeatures = np.zeros((nTrain, dimVectors))
#     trainLabels = np.zeros((nTrain,), dtype=np.int32)


#     for i in xrange(nTrain):
#         words, trainLabels[i] = trainset[i]
#         trainFeatures[i, :] = getMLPSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

#     # Prepare dev set features
#     #devset = dataset.getDevSentences()
#     nDev = len(devset)
#     devFeatures = np.zeros((nDev, dimVectors))
#     devLabels = np.zeros((nDev,), dtype=np.int32)
#     for i in xrange(nDev):
#         words, devLabels[i] = devset[i]
#         devFeatures[i, :] = getMLPSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

#     # Prepare test set features
#     #testset = dataset.getTestSentences()
#     nTest = len(testset)
#     testFeatures = np.zeros((nTest, dimVectors))
#     testLabels = np.zeros((nTest,), dtype=np.int32)
#     for i in xrange(nTest):
#         words, testLabels[i] = testset[i]
#         testFeatures[i, :] = getMLPSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

#     return trainFeatures, trainLabels, devFeatures, devLabels, testFeatures, testLabels, maxLength, dimVectors



# def main(args):
#     """ Train a model to do sentiment analyis"""
#     dataset, tokens, maxSentence = getToxicData()
#     print len(dataset)
#     print dataset[0]
#     print 'hi'
#     print dataset[1]
#     getCharLevelLSTMData()

#     # Shuffle data
#     shuffle(dataset)

#     num_data = len(dataset)

#     # Create train, dev, and test
#     train_cutoff = int(0.6 * num_data)
#     dev_start = int(0.6 * num_data) + 1
#     dev_cutoff = int(0.8 * num_data)

#     trainset = dataset[:train_cutoff]
#     devset = dataset[dev_start:dev_cutoff]
#     testset = dataset[dev_cutoff + 1:]

#     nWords = len(tokens)

#     if args.yourvectors:
#         _, wordVectors, _ = load_saved_params()
#         wordVectors = np.concatenate(
#             (wordVectors[:nWords,:], wordVectors[nWords:,:]),
#             axis=1)
#     elif args.pretrained:
#         wordVectors = glove.loadWordVectors(tokens)
#     dimVectors = wordVectors.shape[1]

#     # Load the train set
#     #trainset = dataset.getTrainSentences()
#     nTrain = len(trainset)
#     trainFeatures = np.zeros((nTrain, dimVectors))
#     trainLabels = np.zeros((nTrain,), dtype=np.int32)


#     for i in xrange(nTrain):
#         words, trainLabels[i] = trainset[i]
#         trainFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

#     # Prepare dev set features
#     #devset = dataset.getDevSentences()
#     nDev = len(devset)
#     devFeatures = np.zeros((nDev, dimVectors))
#     devLabels = np.zeros((nDev,), dtype=np.int32)
#     for i in xrange(nDev):
#         words, devLabels[i] = devset[i]
#         devFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

#     # Prepare test set features
#     #testset = dataset.getTestSentences()
#     nTest = len(testset)
#     testFeatures = np.zeros((nTest, dimVectors))
#     testLabels = np.zeros((nTest,), dtype=np.int32)
#     for i in xrange(nTest):
#         words, testLabels[i] = testset[i]
#         testFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

#     # We will save our results from each run
#     results = []
#     regValues = getRegularizationValues()
#     print "SVM Results:"

#     clf = SVC()
#     clf.fit(trainFeatures, trainLabels)

#     # Test on train set
#     pred = clf.predict(trainFeatures)
#     trainAccuracy = accuracy(trainLabels, pred)
#     print "Train accuracy (%%): %f" % trainAccuracy

#     # Test on dev set
#     pred = clf.predict(devFeatures)
#     devAccuracy = accuracy(devLabels, pred)
#     print "Dev accuracy (%%): %f" % devAccuracy

#     # Test on test set
#     # Note: always running on test is poor style. Typically, you should
#     # do this only after validation.
#     pred = clf.predict(testFeatures)
#     testAccuracy = accuracy(testLabels, pred)
#     print "Test accuracy (%%): %f" % testAccuracy

#     results.append({
#         "reg": 0.0,
#         "clf": clf,
#         "train": trainAccuracy,
#         "dev": devAccuracy,
#         "test": testAccuracy})

#     # Print the accuracies
#     print ""
#     print "=== Recap ==="
#     print "Reg\t\tTrain\tDev\tTest"
#     for result in results:
#         print "%.2E\t%.3f\t%.3f\t%.3f" % (
#             result["reg"],
#             result["train"],
#             result["dev"],
#             result["test"])
#     print ""

#     bestResult = chooseBestModel(results)
#     # print "Best regularization value: %0.2E" % bestResult["reg"]
#     # print "Test accuracy (%%): %f" % bestResult["test"]

#     # do some error analysis
#     if args.pretrained:
#         plotRegVsAccuracy(regValues, results, "q4_reg_v_acc.png")
#         outputConfusionMatrix(devFeatures, devLabels, bestResult["clf"],
#                               "q4_dev_svm_conf.png")
#         outputPredictions(devset, devFeatures, devLabels, bestResult["clf"],
#                           "q4_dev_svm_pred.txt")



# if __name__ == "__main__":
#     main(getArguments())




###########################################################################################

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

def getLSTMSentenceFeatures(tokens, wordVectors, dimVectors, sentence, maxLength):
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
    # - sentVector: feature vector for the sentence\\


    sentVector = np.zeros((maxLength, dimVectors))

    ### YOUR CODE HERE
    for i in range(0, len(sentence)):
        word = sentence[i]
        if i == maxLength:
            break
        if word in tokens:
            sentVector[i] = wordVectors[tokens[word]]

    return sentVector

def getMLPSentenceFeatures(tokens, wordVectors, dimVectors, sentence, maxLength):
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
    # - sentVector: feature vector for the sentence\\


    sentVector = np.zeros((dimVectors,))
    count = 0.0
    ### YOUR CODE HERE
    for i in range(0, len(sentence)):
        word = sentence[i]
        if word in tokens:
            sentVector += wordVectors[tokens[word]]
            count += 1.0
    sentVector = np.divide(sentVector, count)
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

def getMultiLSTMInput(sentence, tokens, wordVectors, maxLength, dimVectors):
    words = [x.lower() for x in re.findall(r"[\w']+|[.,!?;]", sentence)]
    return getLSTMSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)



def getToxicData():
    data = []
    maxSentence = 0
    with open('balanced_data_kevin.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        count = 0
        for i, row in enumerate(reader):
            if count == 0:
                count += 1
                continue
            data.append(([x.lower() for x in re.findall(r"[\w']+|[.,!?;]", row[1])], int(row[2])))

    tokens = dict()
    tokenfreq = dict()
    wordcount = 0
    revtokens = []
    idx = 1

    for row in data:
        sentence = row[0]
        if len(sentence) > maxSentence:
            maxSentence = len(sentence)

        for w in sentence:
            wordcount += 1
            if not w in tokens:
                tokens[w] = idx
                revtokens += [w]
                tokenfreq[w] = 1
                idx += 1
            else:
                tokenfreq[w] += 1

    tokens["UNK"] = 0
    revtokens += ["UNK"]
    tokenfreq["UNK"] = 1
    wordcount += 1

    return data, tokens, maxSentence

def getMLToxicDataFixed():
    data = []
    maxSentence = 0
    max_nontoxic = 100
    with open('balanced_data_kevin.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        count = 0
        for i, row in enumerate(reader):
            if count == 0:
                count += 1
                continue

            if int(row[2]) != 0:
                data.append(([x.lower() for x in re.findall(r"[\w']+|[.,!?;]", row[1])], int(row[3]), int(row[4]), int(row[5]), int(row[6]), int(row[7])))
                count += 1

            if count == 10001:
                break
            # if int(row[2]) == 0 and max_nontoxic != 0:
            #     data.append(([x.lower() for x in re.findall(r"[\w']+|[.,!?;]", row[1])], int(row[3]), int(row[4]), int(row[5]), int(row[6]), int(row[7])))
            #     max_nontoxic = max_nontoxic - 1
            # elif int(row[2]) == 1:
            #     data.append(([x.lower() for x in re.findall(r"[\w']+|[.,!?;]", row[1])], int(row[3]), int(row[4]), int(row[5]), int(row[6]), int(row[7])))

    tokens = dict()
    tokenfreq = dict()
    wordcount = 0
    revtokens = []
    idx = 1

    for row in data:
        sentence = row[0]
        if len(sentence) > maxSentence:
            maxSentence = len(sentence)

        for w in sentence:
            wordcount += 1
            if not w in tokens:
                tokens[w] = idx
                revtokens += [w]
                tokenfreq[w] = 1
                idx += 1
            else:
                tokenfreq[w] += 1

    tokens["UNK"] = 0
    revtokens += ["UNK"]
    tokenfreq["UNK"] = 1
    wordcount += 1

    return data, tokens, maxSentence


def getMLToxicData():
    data = []
    maxSentence = 0
    with open('balanced_data_kevin.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        count = 0
        for i, row in enumerate(reader):
            if count == 0:
                count += 1
                continue
            data.append(([x.lower() for x in re.findall(r"[\w']+|[.,!?;]", row[1])], int(row[3]), int(row[4]), int(row[5]), int(row[6]), int(row[7])))

    tokens = dict()
    tokenfreq = dict()
    wordcount = 0
    revtokens = []
    idx = 1

    for row in data:
        sentence = row[0]
        if len(sentence) > maxSentence:
            maxSentence = len(sentence)

        for w in sentence:
            wordcount += 1
            if not w in tokens:
                tokens[w] = idx
                revtokens += [w]
                tokenfreq[w] = 1
                idx += 1
            else:
                tokenfreq[w] += 1

    tokens["UNK"] = 0
    revtokens += ["UNK"]
    tokenfreq["UNK"] = 1
    wordcount += 1

    return data, tokens, maxSentence

def getMLPMultiData():
    dataset, tokens, maxLength= getMLToxicData()
    # Shuffle data
    shuffle(dataset)

    maxLength = 100

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
    trainLabels = np.zeros((nTrain, 5), dtype=np.int32)


    for i in xrange(nTrain):
        words = trainset[i][0]
        trainLabels[i][0]= trainset[i][1]
        trainLabels[i][1]= trainset[i][2]
        trainLabels[i][2]= trainset[i][3]
        trainLabels[i][3]= trainset[i][4]
        trainLabels[i][4]= trainset[i][5]
        #trainLabels[i][5]= trainset[i][6]
        trainFeatures[i, :] = getMLPSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

    # Prepare dev set features
    #devset = dataset.getDevSentences()
    nDev = len(devset)
    devFeatures = np.zeros((nDev, dimVectors))
    devLabels = np.zeros((nDev, 5), dtype=np.int32)
    for i in xrange(nDev):
        words = devset[i][0]
        devLabels[i][0] = devset[i][1]
        devLabels[i][1] = devset[i][2]
        devLabels[i][2] = devset[i][3]
        devLabels[i][3] = devset[i][4]
        devLabels[i][4] = devset[i][5]
        #devLabels[i][5] = devset[i][6]
        devFeatures[i, :] = getMLPSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

    # Prepare test set features
    #testset = dataset.getTestSentences()
    nTest = len(testset)
    testFeatures = np.zeros((nTest, dimVectors))
    testLabels = np.zeros((nTest, 5), dtype=np.int32)
    for i in xrange(nTest):
        words = testset[i][0]
        testLabels[i][0] = testset[i][1]
        testLabels[i][1] = testset[i][2]
        testLabels[i][2] = testset[i][3]
        testLabels[i][3] = testset[i][4]
        testLabels[i][4] = testset[i][5]
        #testLabels[i][5] = testset[i][6]
        testFeatures[i, :] = getMLPSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

    return trainFeatures, trainLabels, devFeatures, devLabels, testFeatures, testLabels, maxLength, dimVectors


def getLSTMMultiData():
    dataset, tokens, maxLength= getMLToxicData()
    # Shuffle data
    shuffle(dataset)

    maxLength = 100

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
    trainFeatures = np.zeros((nTrain, maxLength, dimVectors))
    trainLabels = np.zeros((nTrain, 5), dtype=np.int32)


    for i in xrange(nTrain):
        words = trainset[i][0]
        trainLabels[i][0]= trainset[i][1]
        trainLabels[i][1]= trainset[i][2]
        trainLabels[i][2]= trainset[i][3]
        trainLabels[i][3]= trainset[i][4]
        trainLabels[i][4]= trainset[i][5]
        #trainLabels[i][5]= trainset[i][6]
        trainFeatures[i, :] = getLSTMSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

    # Prepare dev set features
    #devset = dataset.getDevSentences()
    nDev = len(devset)
    devFeatures = np.zeros((nDev, maxLength, dimVectors))
    devLabels = np.zeros((nDev, 5), dtype=np.int32)
    for i in xrange(nDev):
        words = devset[i][0]
        devLabels[i][0] = devset[i][1]
        devLabels[i][1] = devset[i][2]
        devLabels[i][2] = devset[i][3]
        devLabels[i][3] = devset[i][4]
        devLabels[i][4] = devset[i][5]
        #devLabels[i][5] = devset[i][6]
        devFeatures[i, :] = getLSTMSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

    # Prepare test set features
    #testset = dataset.getTestSentences()
    nTest = len(testset)
    testFeatures = np.zeros((nTest, maxLength, dimVectors))
    testLabels = np.zeros((nTest, 5), dtype=np.int32)
    for i in xrange(nTest):
        words = testset[i][0]
        testLabels[i][0] = testset[i][1]
        testLabels[i][1] = testset[i][2]
        testLabels[i][2] = testset[i][3]
        testLabels[i][3] = testset[i][4]
        testLabels[i][4] = testset[i][5]
        #testLabels[i][5] = testset[i][6]
        testFeatures[i, :] = getLSTMSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

    return trainFeatures, trainLabels, devFeatures, devLabels, testFeatures, testLabels, maxLength, dimVectors#, tokens, wordVectors

def getLSTMData():
    dataset, tokens, maxLength= getToxicData()
    # Shuffle data
    shuffle(dataset)

    maxLength = 100

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
    trainFeatures = np.zeros((nTrain, maxLength, dimVectors))
    trainLabels = np.zeros((nTrain,), dtype=np.int32)


    for i in xrange(nTrain):
        words, trainLabels[i] = trainset[i]
        trainFeatures[i, :] = getLSTMSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

    # Prepare dev set features
    #devset = dataset.getDevSentences()
    nDev = len(devset)
    devFeatures = np.zeros((nDev, maxLength, dimVectors))
    devLabels = np.zeros((nDev,), dtype=np.int32)
    for i in xrange(nDev):
        words, devLabels[i] = devset[i]
        devFeatures[i, :] = getLSTMSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

    # Prepare test set features
    #testset = dataset.getTestSentences()
    nTest = len(testset)
    testFeatures = np.zeros((nTest, maxLength, dimVectors))
    testLabels = np.zeros((nTest,), dtype=np.int32)
    for i in xrange(nTest):
        words, testLabels[i] = testset[i]
        testFeatures[i, :] = getLSTMSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

    return trainFeatures, trainLabels, devFeatures, devLabels, testFeatures, testLabels, maxLength, dimVectors

def getMLPData():
    dataset, tokens, maxLength= getToxicData()
    # Shuffle data
    shuffle(dataset)

    maxLength = 100

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
    trainLabels = np.zeros((nTrain,), dtype=np.int32)


    for i in xrange(nTrain):
        words, trainLabels[i] = trainset[i]
        trainFeatures[i, :] = getMLPSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

    # Prepare dev set features
    #devset = dataset.getDevSentences()
    nDev = len(devset)
    devFeatures = np.zeros((nDev, dimVectors))
    devLabels = np.zeros((nDev,), dtype=np.int32)
    for i in xrange(nDev):
        words, devLabels[i] = devset[i]
        devFeatures[i, :] = getMLPSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

    # Prepare test set features
    #testset = dataset.getTestSentences()
    nTest = len(testset)
    testFeatures = np.zeros((nTest, dimVectors))
    testLabels = np.zeros((nTest,), dtype=np.int32)
    for i in xrange(nTest):
        words, testLabels[i] = testset[i]
        testFeatures[i, :] = getMLPSentenceFeatures(tokens, wordVectors, dimVectors, words, maxLength)

    return trainFeatures, trainLabels, devFeatures, devLabels, testFeatures, testLabels, maxLength, dimVectors


def getCharLevelLSTMData():
    
    dataset, tokens, maxLength= getToxicData()
    # Shuffle data
    shuffle(dataset)

    maxLength =300
    num_data = len(dataset)

    data = []
    maxSentence = 0
    with open('balanced_data_kevin.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        count = 0
        for i, row in enumerate(reader):
            if count == 0:
                count += 1
                continue
            data.append(' '.join(([x.lower() for x in re.findall(r"[\w']+|[.,!?;]", row[1])])))


    raw_text =  ' '.join([data[i] for i in range(num_data)])
    chars = sorted(list(set(raw_text)))

    char_to_int = dict((c, i+1) for i, c in enumerate(chars))
    int_to_char = dict((i+1, c) for i, c in enumerate(chars))

    vocab_size = len(chars)
    # print ('CHARTOINT IS ', char_to_int)
    # print ('INT TO CAHR' , int_to_char)

    data_ = []
    for i in range(num_data):

        #print i, data[i]
        #data_.append(([data[char_to_int[char]] for char in data[i][0]],data[i][1]))
        data_.append(([char_to_int[char] for char in data[i]],dataset[i][1]))
    
    # print data_[0]
    # print 'final'
    #print data_[1]
    # Create train, dev, and test
    train_cutoff = int(0.6 * num_data)
    dev_start = int(0.6 * num_data) + 1
    dev_cutoff = int(0.8 * num_data)

    trainset = data_[:train_cutoff]
    devset = data_[dev_start:dev_cutoff]
    testset = data_[dev_cutoff + 1:]

    #took from here
    dimVectors = len(char_to_int)

    # Load the train set
    #trainset = dataset.getTrainSentences()
    nTrain = len(trainset)
    trainFeatures = np.zeros((nTrain, maxLength, dimVectors))
    trainLabels = np.zeros((nTrain,), dtype=np.int32)


    for i in xrange(nTrain):
    #for i in xrange(1):
        words, trainLabels[i] = trainset[i]
        # print ('words :',words)
        trainFeatures[i, :] = getChar_level_LSTMSentenceFeatures(int_to_char, dimVectors, words, maxLength)
        # print (' trainFeatures  are " ', trainFeatures[i,:])

    
    # print trainset[0]

    nDev = len(devset)
    devFeatures = np.zeros((nDev, maxLength, dimVectors))
    devLabels = np.zeros((nDev,), dtype=np.int32)
    for i in xrange(nDev):
        words, devLabels[i] = devset[i]
        devFeatures[i, :] = getChar_level_LSTMSentenceFeatures(int_to_char, dimVectors, words, maxLength)

    # Prepare test set features
    #testset = dataset.getTestSentences()
    nTest = len(testset)
    testFeatures = np.zeros((nTest, maxLength, dimVectors))
    testLabels = np.zeros((nTest,), dtype=np.int32)
    for i in xrange(nTest):
        words, testLabels[i] = testset[i]
        testFeatures[i, :] = getChar_level_LSTMSentenceFeatures(int_to_char, dimVectors, words, maxLength)

    #print 'Hi nad Fi'
    #print trainFeatures[1][1]


    return trainFeatures, trainLabels, devFeatures, devLabels, testFeatures, testLabels, maxLength, dimVectors


def getChar_level_LSTMSentenceFeatures(tokens, dimVectors, sentence, maxLength):
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
    # - sentVector: feature vector for the sentence\\


    sentVector = np.zeros((maxLength, dimVectors))

    ### YOUR CODE HERE
    
    sentence_len = len(sentence)
    for i in range(0, sentence_len):
        char = sentence[i]
        if i == maxLength:
            break
        if char in tokens.keys():
            #sentVector[i] += wordVectors[tokens[word]]

            #sentVector[i][tokens[char]-1] += 1
            sentVector[i][char -1] +=1 
    return sentVector

def main(args):
    """ Train a model to do sentiment analyis"""
    dataset, tokens, maxSentence = getToxicData()
    print len(dataset)
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

    if args.yourvectors:
        _, wordVectors, _ = load_saved_params()
        wordVectors = np.concatenate(
            (wordVectors[:nWords,:], wordVectors[nWords:,:]),
            axis=1)
    elif args.pretrained:
        wordVectors = glove.loadWordVectors(tokens)
    dimVectors = wordVectors.shape[1]

    # Load the train set
    #trainset = dataset.getTrainSentences()
    nTrain = len(trainset)
    trainFeatures = np.zeros((nTrain, dimVectors))
    trainLabels = np.zeros((nTrain,), dtype=np.int32)


    for i in xrange(nTrain):
        words, trainLabels[i] = trainset[i]
        trainFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

    # Prepare dev set features
    #devset = dataset.getDevSentences()
    nDev = len(devset)
    devFeatures = np.zeros((nDev, dimVectors))
    devLabels = np.zeros((nDev,), dtype=np.int32)
    for i in xrange(nDev):
        words, devLabels[i] = devset[i]
        devFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

    # Prepare test set features
    #testset = dataset.getTestSentences()
    nTest = len(testset)
    testFeatures = np.zeros((nTest, dimVectors))
    testLabels = np.zeros((nTest,), dtype=np.int32)
    for i in xrange(nTest):
        words, testLabels[i] = testset[i]
        testFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

    # We will save our results from each run
    results = []
    regValues = getRegularizationValues()
    print "SVM Results:"

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

