import numpy as np

import MLCourse.dataloader as dtl
import MLCourse.utilities as utils
import classalgorithms as algs

def getaccuracy(ytest, predictions):
    correct = 0
    # count number of correct predictions
    correct = np.sum(ytest == predictions)
    # return percent correct
    return (correct / float(len(ytest))) * 100

def geterror(ytest, predictions):
    return (100 - getaccuracy(ytest, predictions))

""" k-fold cross-validation
K - number of folds
X - data to partition
Y - targets to partition
Algorithm - the algorithm class to instantiate
parameters - a list of parameter dictionaries to test

NOTE: utils.leaveOneOut will likely be useful for this problem.
Check utilities.py for example usage.
"""
def cross_validate(K, X, Y, Algorithm, parameters):
    all_errors = np.zeros((len(parameters), K))

    for k in range(K):
        # Split the dataset to train set and test set according to number of folds k
        X_n = np.split(X, K)
        Y_n = np.split(Y, K)

        Xtest = X_n.pop(k)
        Ytest = Y_n.pop(k)
        Xtrain = np.concatenate(X_n)
        Ytrain = np.concatenate(Y_n)

        for i, params in enumerate(parameters):
            learner = Algorithm (params)
            # Train model
            learner.learn(Xtrain, Ytrain)
            # Test model
            predictions = learner.predict(Xtest)
            error = geterror(Ytest, predictions)
            all_errors[i, k] = error

    avg_errors = np.mean(all_errors, axis=1)
    best_error = np.inf

    for i, params in enumerate(parameters):
        print('Cross validate parameters:', params)
        print('average error:', avg_errors[i])
        if avg_errors[i] < best_error:
            best_error = avg_errors[i]
            best_parameters = parameters[i]

    return best_parameters

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Arguments for running.')
    parser.add_argument('--trainsize', type=int, default=5000,
                        help='Specify the train set size')
    parser.add_argument('--testsize', type=int, default=5000,
                        help='Specify the test set size')
    parser.add_argument('--numruns', type=int, default=1,
                        help='Specify the number of runs')
    parser.add_argument('--dataset', type=str, default="susy",
                        help='Specify the name of the dataset')
    # Added an extra argument for number of folds for debugging
    parser.add_argument('--numfolds', type=int, default=2,
                        help='Specify the number of folds')

    args = parser.parse_args()
    trainsize = args.trainsize
    testsize = args.testsize
    numruns = args.numruns
    dataset = args.dataset
    """ Uncomment the nextline if want to test KernelLogisticRegressionHamming on censes dataset """
    # dataset = 'census'
    numfolds = args.numfolds

    classalgs = {
        # 'Random': algs.Classifier,
        'Naive Bayes': algs.NaiveBayes,
        # 'Linear Regression': algs.LinearRegressionClass,
        'Logistic Regression': algs.LogisticReg,
        'Neural Network': algs.NeuralNet,
        'Kernel Logistic Regression': algs.KernelLogisticRegression,
        # 'KernelLogisticRegressionHamming': algs.KernelLogisticRegressionHamming,
    }
    numalgs = len(classalgs)

    # Specify the name of the algorithm and an array of parameter values to try
    # if an algorithm is not include, will run with default parameters
    parameters = {
        # name of the algorithm to run
        'Naive Bayes': [
            # first set of parameters to try
            { 'usecolumnones': True },
            # second set of parameters to try
            { 'usecolumnones': False },
        ],
        'Logistic Regression': [
            { 'stepsize': 0.001 },
            { 'stepsize': 0.01 },
            { 'stepsize': 0.1 },
        ],
        'Neural Network': [
            { 'epochs': 100, 'nh': 4 },
            { 'epochs': 100, 'nh': 8 },
            { 'epochs': 100, 'nh': 16 },
            { 'epochs': 100, 'nh': 32 },
        ],
        'Kernel Logistic Regression': [
            { 'centers': 10, 'stepsize': 0.01 },
            { 'centers': 20, 'stepsize': 0.01 },
            { 'centers': 40, 'stepsize': 0.01 },
            { 'centers': 80, 'stepsize': 0.01 },
        ],
        'KernelLogisticRegressionHamming': [
            { 'centers': 10, 'stepsize': 0.01 },
            { 'centers': 20, 'stepsize': 0.01 },
            { 'centers': 40, 'stepsize': 0.01 },
            { 'centers': 80, 'stepsize': 0.01 },
        ]
    }

    # initialize the errors for each parameter setting to 0
    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros(numruns)
        
    for r in range(numruns):
        if dataset == "susy":
            trainset, testset = dtl.load_susy(trainsize, testsize)
        elif dataset == "census":
            trainset, testset = dtl.load_census(trainsize,testsize)
        else:
            raise ValueError("dataset %s unknown" % dataset)

        # print(trainset[0])
        Xtrain = trainset[0]
        Ytrain = trainset[1]
        # cast the Y vector as a matrix
        Ytrain = np.reshape(Ytrain, [len(Ytrain), 1])
        Xtest = testset[0]
        Ytest = testset[1]
        # cast the Y vector as a matrix
        Ytest = np.reshape(Ytest, [len(Ytest), 1])

        best_parameters = {}

        for learnername, Learner in classalgs.items():
            params = parameters.get(learnername, [ None ])
            print('\n\nRunning learner = ' + learnername + '\tRun = ' + str(r))
            best_parameters[learnername] = cross_validate(numfolds, Xtrain, Ytrain, Learner, params)

        print('\n')
        for learnername, Learner in classalgs.items():
            params = best_parameters[learnername]
            print('Best Parameter for ' + learnername + 'is '+ str(params))
            learner = Learner(params)
            # Train model
            learner.learn(Xtrain, Ytrain)
            # Test model
            pred = learner.predict(Xtest)
            error = geterror(Ytest, pred)
            errors[learnername][r] = error

    print('\n' * 2)
    for learnername in classalgs:
        aveerror = np.mean(errors[learnername])
        stderror = np.std(errors[learnername]) / np.sqrt(numruns)
        print('Average error for ' + learnername + ': ' + str(aveerror))
        print('Standard error for ' + learnername + ': ' + str(stderror))
