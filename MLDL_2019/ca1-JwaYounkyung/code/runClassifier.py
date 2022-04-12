"""
This module is for training, testing an evaluating classifiers.
"""

from numpy import *
from pylab import *

import sys
import util
import binary

def trainTest(classifier, X, Y, Xtest, Ytest):
    """
    Train a classifier on data (X,Y) and evaluate on
    data (Xtest,Ytest).  Return a triple of:
      * Training data accuracy
      * Test data accuracy
      * Individual predictions on Xtest.
    """

    classifier.reset()                           # initialize the classifier
    classifier.train(X, Y);                      # train it

    Ypred = classifier.predictAll(X);               # predict the training data
    trAcc = mean((Y     >= 0) == (Ypred >= 0));     # check to see how often the predictions are right

    Ypred = classifier.predictAll(Xtest);           # predict the training data
    teAcc = mean((Ytest >= 0) == (Ypred >= 0));     # check to see how often the predictions are right

    print "Training accuracy %g, test accuracy %g" % (trAcc, teAcc)

    return (trAcc, teAcc, Ypred)

def trainTestSet(classifier, dataset):
    trainTest(classifier, dataset.X, dataset.Y, dataset.Xte, dataset.Yte)

def learningCurve(classifier, X, Y, Xtest, Ytest):
    """
    Generate a learning curve by repeatedly halving the amount of
    training data until none is left.

    We return a triple containing:
      * The sizes of data sets we trained on
      * The training accuracies at each level
      * The test accuracies at each level
    """

    N = X.shape[0]             # how many total points?
    M = int(ceil(log2(N)))     # how many classifiers will we have to train?

    dataSizes = zeros(M)
    trainAcc  = zeros(M)
    testAcc   = zeros(M)
    
    for i in range(1, M+1):    # loop over "skip lengths"
        # select every 2^(M-i)th point
        ids = arange(0, N, 2**(M-i))
        Xtr = X[ids, :]
        Ytr = Y[ids]

        # report what we're doing
        print "Training classifier on %d points..." % ids.size

        # train the classifier
        (trAcc, teAcc, Ypred) = trainTest(classifier, Xtr, Ytr, Xtest, Ytest)
        
        # store the results
        dataSizes[i-1] = ids.size
        trainAcc[i-1]  = trAcc
        testAcc[i-1]   = teAcc

    return (dataSizes, trainAcc, testAcc)

def learningCurveSet(classifier, dataset):
    return learningCurve(classifier, dataset.X, dataset.Y, dataset.Xte, dataset.Yte)



def hyperparamCurve(classifier, hpName, hpValues, X, Y, Xtest, Ytest):
    M = len(hpValues)
    trainAcc = zeros(M)
    testAcc  = zeros(M)
    for m in range(M):
        # report what we're doing
        print "Training classifier with %s=%g..." % (hpName, hpValues[m])
        
        # train the classifier
        classifier.setOption(hpName, hpValues[m])
        classifier.reset()
        (trAcc, teAcc, Ypred) = trainTest(classifier, X, Y, Xtest, Ytest)

        # store the results
        trainAcc[m] = trAcc
        testAcc[m]  = teAcc

    return (hpValues, trainAcc, testAcc)

def hyperparamCurveSet(classifier, hpName, hpValues, dataset):
    return hyperparamCurve(classifier, hpName, hpValues, dataset.X, dataset.Y, dataset.Xte, dataset.Yte)

def plotCurve(titleString, res):
    plot(res[0], res[1], 'b-',
         res[0], res[2], 'r-')
    legend( ('Train', 'Test') )
    ylabel('Accuracy')
    title(titleString)
    show()

def shufflePoints(X, Y):
    """
    Randomize the order of the points.
    """

    [N,D] = X.shape
    order = range(N)
    util.permute(order)

    retX = X[order,:]
    retY = Y[order]
    return (retX, retY)
            

def plotData(X, Y):
    plot(X[Y>=0,0], X[Y>=0,1], 'bo',
         X[Y< 0,0], X[Y< 0,1], 'rx')
    legend( ('+1', '-1') )
    show()

def plotClassifier(w, b):
    axes = figure(1).get_axes()[0]
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()

    xmin = xlim[0] + (xlim[1] - xlim[0]) / 100
    xmax = xlim[1] - (xlim[1] - xlim[0]) / 100
    ymin = ylim[0] + (ylim[1] - ylim[0]) / 100
    ymax = ylim[1] - (ylim[1] - ylim[0]) / 100

    # find the zeros along each axis
    # w0*l + w1*? + b = 0  ==>  ? = -(b + w0*l) / w1
    xmin_zero = - (b + w[0] * xmin) / w[1]
    xmax_zero = - (b + w[0] * xmax) / w[1]
    ymin_zero = - (b + w[1] * ymin) / w[0]
    ymax_zero = - (b + w[1] * ymax) / w[0]

    # now, two of these should actually be in bounds, figure out which
    inBounds = []
    if ylim[0] <= xmin_zero and xmin_zero <= ylim[1]:
        inBounds.append( (xmin, xmin_zero) )
    if ylim[0] <= xmax_zero and xmax_zero <= ylim[1]:
        inBounds.append( (xmax, xmax_zero) )
    if xlim[0] <= ymin_zero and ymin_zero <= xlim[1]:
        inBounds.append( (ymin_zero, ymin) )
    if xlim[0] <= ymax_zero and ymax_zero <= xlim[1]:
        inBounds.append( (ymax_zero, ymax) )

    plot( array([inBounds[0][0], inBounds[1][0]]), array([inBounds[0][1], inBounds[1][1]]), 'g-', linewidth=2 )
    figure(1).set_axes([axes])
    
def dumpMegamFormat(fname, Xtr, Ytr, Xte, Yte):
    def writeIt(f, X, Y):
        N,D = X.shape
        for n in range(N):
            f.write(str(Y[n]))
            for d in range(D):
                if X[n,d] != 0:
                    f.write(" f" + str(d) + " " + str(X[n,d]))
            f.write("\n")

    f = open(fname, 'w')
    writeIt(f, Xtr, Ytr)
    f.write("TEST\n")
    writeIt(f, Xte, Yte)
    f.close()

def dumpMegamFormatSet(fname, dataset):
    dumpMegamFormat(fname, dataset.X, dataset.Y, dataset.Xte, dataset.Yte)

def dumpSVMFormat(fname, Xtr, Ytr, Xte, Yte):
    def writeIt(f, X, Y):
        N,D = X.shape
        for n in range(N):
            f.write(str(Y[n]))
            for d in range(D):
                if X[n,d] != 0:
                    f.write(" " + str(d+1) + ":" + str(X[n,d]))
            f.write("\n")

    f = open(fname, 'w')
    writeIt(f, Xtr, Ytr)
    writeIt(f, Xte, Yte)
    f.close()


def dumpSVMFormatSet(fname, dataset):
    dumpSVMFormat(fname, dataset.X, dataset.Y, dataset.Xte, dataset.Yte)

def trainPrunedTest(classifier, X, Y, Xtest, Ytest):
    """
    Train a classifier on data (X,Y) and evaluate on
    data (Xtest,Ytest).  Return a triple of:
      * Training data accuracy
      * Test data accuracy
      * Individual predictions on Xtest.
    """

    classifier.reset()                           # initialize the classifier
    classifier.train(X, Y);                      # train it

    classifier.pruneByChiSquare(0.05)

    Ypred = classifier.predictAll(X);               # predict the training data
    trAcc = mean((Y     >= 0) == (Ypred >= 0));     # check to see how often the predictions are right

    Ypred = classifier.predictAll(Xtest);           # predict the training data
    teAcc = mean((Ytest >= 0) == (Ypred >= 0));     # check to see how often the predictions are right

    print "Training accuracy %g, test accuracy %g" % (trAcc, teAcc)

    return (trAcc, teAcc, Ypred)

def learningPrunedCurve(classifier, X, Y, Xtest, Ytest):
    """
    Generate a learning curve by repeatedly halving the amount of
    training data until none is left.

    We return a triple containing:
      * The sizes of data sets we trained on
      * The training accuracies at each level
      * The test accuracies at each level
    """

    N = X.shape[0]             # how many total points?
    M = int(ceil(log2(N)))     # how many classifiers will we have to train?

    dataSizes = zeros(M)
    trainAcc  = zeros(M)
    testAcc   = zeros(M)
    
    for i in range(1, M+1):    # loop over "skip lengths"
        # select every 2^(M-i)th point
        ids = arange(0, N, 2**(M-i))
        Xtr = X[ids, :]
        Ytr = Y[ids]

        # report what we're doing
        print "Training classifier on %d points..." % ids.size

        # train the classifier
        (trAcc, teAcc, Ypred) = trainPrunedTest(classifier, Xtr, Ytr, Xtest, Ytest)
        
        # store the results
        dataSizes[i-1] = ids.size
        trainAcc[i-1]  = trAcc
        testAcc[i-1]   = teAcc

    return (dataSizes, trainAcc, testAcc)

def learningPrunedCurveSet(classifier, dataset):
    return learningPrunedCurve(classifier, dataset.X, dataset.Y, dataset.Xte, dataset.Yte)

def hyperparamPCurve(classifier, hpName, hpValues, X, Y, Xtest, Ytest):
    M = len(hpValues)
    trainAcc = zeros(M)
    testAcc  = zeros(M)
    for m in range(M):
        # report what we're doing
        print "Training classifier with %s=%g..." % (hpName, hpValues[m])
        
        # train the classifier
        classifier.reset()

        maxPvalue = hpValues[m]

        (trAcc, teAcc, Ypred) = trainPvalueTest(classifier, maxPvalue, X, Y, Xtest, Ytest)

        # store the results
        trainAcc[m] = trAcc
        testAcc[m]  = teAcc

    return (hpValues, trainAcc, testAcc)

def hyperparamPCurveSet(classifier, hpName, hpValues, dataset):
    return hyperparamPCurve(classifier, hpName, hpValues, dataset.X, dataset.Y, dataset.Xte, dataset.Yte)

def trainPvalueTest(classifier, maxPvalue, X, Y, Xtest, Ytest):
    """
    Train a classifier on data (X,Y) and evaluate on
    data (Xtest,Ytest).  Return a triple of:
      * Training data accuracy
      * Test data accuracy
      * Individual predictions on Xtest.
    """

    classifier.reset()                           # initialize the classifier
    classifier.train(X, Y);                      # train it

    classifier.pruneByChiSquare(maxPvalue)

    Ypred = classifier.predictAll(X);               # predict the training data
    trAcc = mean((Y     >= 0) == (Ypred >= 0));     # check to see how often the predictions are right

    Ypred = classifier.predictAll(Xtest);           # predict the training data
    teAcc = mean((Ytest >= 0) == (Ypred >= 0));     # check to see how often the predictions are right

    print "Training accuracy %g, test accuracy %g" % (trAcc, teAcc)

    return (trAcc, teAcc, Ypred)


def comparativeTestingCurve(classifier, X, Y, Xtest, Ytest):
    """
    Generate a learning curve by repeatedly halving the amount of
    training data until none is left.

    We return a triple containing:
      * The sizes of data sets we trained on
      * The test accuracies at each level
      * The pruned test accuracies at each level
    """

    N = X.shape[0]             # how many total points?
    M = int(ceil(log2(N)))     # how many classifiers will we have to train?

    dataSizes = zeros(M)
    trainAcc  = zeros(M)
    testAcc   = zeros(M)

    dataSizes2 = zeros(M)
    trainAcc2  = zeros(M)
    testAcc2   = zeros(M)
    
    for i in range(1, M+1):    # loop over "skip lengths"
        # select every 2^(M-i)th point
        ids = arange(0, N, 2**(M-i))
        Xtr = X[ids, :]
        Ytr = Y[ids]

        # report what we're doing
        print "Training classifier on %d points..." % ids.size

        # train the classifier
        (trAcc, teAcc, Ypred) = trainTest(classifier, Xtr, Ytr, Xtest, Ytest)
        (trAcc2, teAcc2, Ypred2) = trainPrunedTest(classifier, Xtr, Ytr, Xtest, Ytest)
        
        # store the results
        dataSizes[i-1] = ids.size
        testAcc2[i-1]  = teAcc2
        testAcc[i-1]   = teAcc

    return (dataSizes, testAcc, testAcc2)

def comparativeTestingCurveSet(classifier, dataset):
    return comparativeTestingCurve(classifier, dataset.X, dataset.Y, dataset.Xte, dataset.Yte)

def comparativeTrainingCurve(classifier, X, Y, Xtest, Ytest):
    """
    Generate a learning curve by repeatedly halving the amount of
    training data until none is left.

    We return a triple containing:
      * The sizes of data sets we trained on
      * The training accuracies at each level
      * The pruned training accuracies at each level
    """

    N = X.shape[0]             # how many total points?
    M = int(ceil(log2(N)))     # how many classifiers will we have to train?

    dataSizes = zeros(M)
    trainAcc  = zeros(M)
    testAcc   = zeros(M)

    dataSizes2 = zeros(M)
    trainAcc2  = zeros(M)
    testAcc2   = zeros(M)
    
    for i in range(1, M+1):    # loop over "skip lengths"
        # select every 2^(M-i)th point
        ids = arange(0, N, 2**(M-i))
        Xtr = X[ids, :]
        Ytr = Y[ids]

        # report what we're doing
        print "Training classifier on %d points..." % ids.size

        # train the classifier
        (trAcc, teAcc, Ypred) = trainTest(classifier, Xtr, Ytr, Xtest, Ytest)
        (trAcc2, teAcc2, Ypred2) = trainPrunedTest(classifier, Xtr, Ytr, Xtest, Ytest)
        
        # store the results
        dataSizes[i-1] = ids.size
        trainAcc2[i-1]  = trAcc2
        trainAcc[i-1]   = trAcc

    return (dataSizes, trainAcc, trainAcc2)

def comparativeTrainingCurveSet(classifier, dataset):
    return comparativeTrainingCurve(classifier, dataset.X, dataset.Y, dataset.Xte, dataset.Yte)
