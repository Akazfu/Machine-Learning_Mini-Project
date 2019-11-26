import numpy as np

import MLCourse.utilities as utils

# Susy: ~50 error
class Classifier:
    def __init__(self, parameters = {}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the training data """
        pass

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        # ytest = ytest.flatten()
        return ytest

# Susy: ~27 error
class LinearRegressionClass(Classifier):
    def __init__(self, parameters = {}):
        self.params = {'regwgt': 0.01}
        self.weights = None

    def learn(self, X, y):
        # Ensure y is {-1,1}
        y = np.copy(y)
        y[y == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = X.shape[0]
        numfeatures = X.shape[1]

        inner = (X.T.dot(X) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.inv(inner).dot(X.T).dot(y) / numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

# Susy: ~25 error
class NaiveBayes(Classifier):
    def __init__(self, parameters = {}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.params = utils.update_dictionary_items({'usecolumnones': False}, parameters)

    def learn(self, Xtrain, ytrain):
        # obtain number of classes
        if ytrain.shape[1] == 1:
            self.numclasses = 2
            self.numsamples = ytrain.shape[0]
            # print(self.numsamples)
            # If usecolumnones is False, it ignores this last feature
            if self.params['usecolumnones'] == False:
                Xtrain = Xtrain[:, :8].copy()
            self.numfeatures = Xtrain.shape[1]
            # print(self.numfeatures)

            count = np.zeros((2,), dtype=int)
            sample_class = [[], []]

            # Compute prior P(c) = 0 and P(c) = 1 from the dataset if it is there
            for i in range (self.numsamples):
                if ytrain[i] == 1:
                    count[1] += 1
                else:
                    count[0] += 1
            self.p_0 = count[0]/self.numsamples
            self.p_1 = count[1]/self.numsamples
            self.mean = np.zeros((self.numclasses, self.numfeatures))
            self.std = np.zeros((self.numclasses, self.numfeatures))

            for i in range (self.numsamples):
                if ytrain[i] == 1:
                    sample_class[1].append(Xtrain[i])
                elif ytrain[i] == 0:
                    sample_class[0].append(Xtrain[i])
            
            for c in range(self.numclasses):
                for j in range (self.numfeatures):
                    mean = []
                    for i in range(count[c]):
                        mean.append(sample_class[c][i][j])
                    self.mean[c][j] = np.mean(mean)
                    self.std[c][j] = np.std(mean)                
        else:
            raise Exception('Can only handle binary classification')

    def predict(self, Xtest):
        numtest = Xtest.shape[0]
        predictions = []
        pred = np.ones((self.numclasses, numtest))

        for i in range (numtest):
            # print('self.numfeatures is '+str(self.numfeatures))
            for j in range (self.numfeatures):
                if self.std[0][j] == 0:
                    pred[0][i] = pred[0][i] * 1
                else:
                    pred[0][i] = pred[0][i] * (1.0/np.sqrt(2*np.pi*(self.std[0][j]**2))) * np.exp(-1.0*np.square(Xtest[i][j]-self.mean[0][j])/(2*(self.std[0][j]**2)))
                
                if self.std[1][j] == 0:
                    pred[1][i] = pred[1][i] * 1
                else:
                    pred[1][i] = pred[1][i] * (1.0/np.sqrt(2*np.pi*(self.std[1][j]**2))) * np.exp(-1.0*np.square(Xtest[i][j]-self.mean[1][j])/(2*(self.std[1][j]**2)))

            pred[0][i] = pred[0][i] * self.p_0
            pred[1][i] = pred[1][i] * self.p_1
       
        for i in range (numtest):
            if pred[1][i] >= pred[0][i]:
                predictions.insert(i, 1)
            else:
                predictions.insert(i, 0)

        return np.reshape(predictions, [numtest, 1])

# Susy: ~23 error
class LogisticReg(Classifier):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({'stepsize': 0.01, 'epochs': 100}, parameters)
        self.weights = None

    def learn(self, X, y):
        self.weights = np.zeros(X.shape[1],)
        epochs = self.params["epochs"]
        stepSize = self.params["stepsize"]
        for j in range(epochs):
            for i in range(X.shape[0]):
                sigmoid = utils.sigmoid(X[i] @ self.weights)
                # print (sigmoid)
                self.weights -= stepSize * (sigmoid-y[i]) * X[i]

    def predict(self, Xtest):
        predictions = []
        numtest = Xtest.shape[0]

        for i in range(numtest):
            prob = utils.sigmoid(np.dot(Xtest, self.weights))
            if  prob[i] >=0.5:
                predictions.insert(i, 1)
            else:
                predictions.insert(i, 0)
      
        return np.reshape(predictions, [numtest, 1])


# Susy: ~23 error (4 hidden units)
class NeuralNet(Classifier):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'nh': 4,
            'transfer': 'sigmoid',
            'stepsize': 0.01,
            'epochs': 10,
        }, parameters)

        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')

        # self.wi = None
        # self.wo = None
    
    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        numfeatures = Xtrain.shape[1]
        epochs = self.params["epochs"]

        self.wi = np.random.randn(self.params['nh'], numfeatures)
        self.wo = np.random.randn(1, self.params['nh'])

        for epoch in range(epochs):
            for i in range(numsamples):
                xtrain = np.reshape(Xtrain[i],(-1,1))
                # Hidden Layer 1
                h1 = self.transfer(np.dot(self.wi,xtrain))
                h1 = np.reshape(h1,(-1,1))
                predicted = self.transfer(np.dot(self.wo,h1))
                
                # Calculate the gradients
                delta = (-ytrain[i]*(1-predicted)) + (1-ytrain[i]) * predicted
                grad_wo = np.dot(delta,h1.T)
                grad_int = (self.wo*h1.T*(1-h1.T)).T
                xtrain = np.reshape(Xtrain[i].T,(-1,1))
                grad_wi = delta*np.dot(grad_int,xtrain.T)

                # Call update function to update the weights
                self.update(grad_wi, grad_wo)

    def predict(self,Xtest):
        predictions = []
        numtest = Xtest.shape[0]
        
        for i in range(0,numtest):
            # Retrieve classes label from (ah, ao)
            ao = self.evaluate(Xtest[i])[1]
            if ao > 0.5:
               predictions.insert(i, 1)
            else:
               predictions.insert(i, 0)
               
        return np.reshape(predictions, [numtest, 1])

    def evaluate(self, inputs):
        # hidden activations
        ah = self.transfer(np.dot(self.wi,inputs.T))

        # output activations
        ao = self.transfer(np.dot(self.wo,ah)).T

        return (
            ah, # shape: [nh, samples]
            ao, # shape: [classes, nh]
        )

    def update(self, inputs, outputs):
        self.wi= self.wi - self.params['stepsize'] * inputs
        self.wo=  self.wo - self.params['stepsize'] * outputs

# Note: high variance in errors! Make sure to run multiple times
# Susy: ~28 error (40 centers)
class KernelLogisticRegression(LogisticReg):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'stepsize': 0.01,
            'epochs': 100,
            'centers': 10,
        }, parameters)
        # self.weights = None

    def learn(self, X, y):
        centers = self.params['centers']
        epochs = self.params["epochs"]
        stepSize = self.params["stepsize"]
        self.centerlist = []

        for c in range(centers):
            index = np.random.choice(X.shape[0])
            self.centerlist.append(X[index])
        
        #initialize kernel as n by k zero matrix 
        kernel = np.zeros((X.shape[0], self.params['centers']))
        for i in range (kernel.shape[0]):
            for j in range(kernel.shape[1]):
                kernel[i][j] = np.dot(X[i],self.centerlist[j])

        self.weights = np.zeros(kernel.shape[1],)
        for epoch in range(epochs):
            for i in range(kernel.shape[0]):
                sigmoid = utils.sigmoid(kernel[i] @ self.weights)
                # print(sigmoid)
                self.weights -= stepSize * (sigmoid-y[i]) * kernel[i]

    def predict(self, Xtest):
        predictions = []
        numtest = Xtest.shape[0]

        Ktest = np.zeros((numtest, self.params['centers']))
        for n in range(numtest):
            for m in range(len(self.centerlist)):
                # print(Ktest[n])
                # print(Xtest[n])
                # print(self.centerlist[m])
                Ktest[n][m] = np.dot(Xtest[n], self.centerlist[m])

        newtest = np.dot(Ktest, self.weights)
        for i in range(newtest.shape[0]):
            prob = utils.sigmoid(newtest)
            if  prob[i] >=0.5:
                predictions.insert(i, 1)
            else:
                predictions.insert(i, 0)
      
        return np.reshape(predictions, [numtest, 1])


# Question 2 b) To test the KernelLogisticRegressionHamming ==> Uncomment the censes dataset line at script_classify.py
class KernelLogisticRegressionHamming(LogisticReg):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'stepsize': 0.01,
            'epochs': 100,
            'centers': 10,
        }, parameters)
        # self.weights = None

    # A simple Implementation for compute the hamming distance
    def hamming(self, x, c):
        distance = 0
        for i in range(len(x)):
            # if two arrays are not identical distance +1
            if x[i] != c[i]:
                distance += 1
        return distance

    def learn(self, X, y):
        centers = self.params['centers']
        epochs = self.params["epochs"]
        stepSize = self.params["stepsize"]
        self.centerlist = []

        for c in range(centers):
            index = np.random.choice(X.shape[0])
            self.centerlist.append(X[index])
        
        #initialize kernel as n by k zero matrix 
        kernel = np.zeros((X.shape[0], self.params['centers']))
        for i in range (kernel.shape[0]):
            for j in range(kernel.shape[1]):
                kernel[i][j] = self.hamming(X[i],self.centerlist[j])

        self.weights = np.zeros(kernel.shape[1],)
        for epoch in range(epochs):
            for i in range(kernel.shape[0]):
                sigmoid = utils.sigmoid(kernel[i] @ self.weights)
                # print(sigmoid)
                self.weights -= stepSize * (sigmoid-y[i]) * kernel[i]

    def predict(self, Xtest):
        predictions = []
        numtest = Xtest.shape[0]

        Ktest = np.zeros((numtest, self.params['centers']))
        for n in range(numtest):
            for m in range(len(self.centerlist)):
                # print(Ktest[n])
                # print(Xtest[n])
                # print(self.centerlist[m])
                Ktest[n][m] = self.hamming(Xtest[n], self.centerlist[m])

        newtest = np.dot(Ktest, self.weights)
        for i in range(newtest.shape[0]):
            prob = utils.sigmoid(newtest)
            if  prob[i] >=0.5:
                predictions.insert(i, 1)
            else:
                predictions.insert(i, 0)
      
        return np.reshape(predictions, [numtest, 1])

# Bonus a) neural network of two hidden layers
class NeuralNet2Hidden(Classifier):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'nh': 4,
            'transfer': 'sigmoid',
            'stepsize': 0.01,
            'epochs': 10,
        }, parameters)

        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')

        # self.wi = None
        # self.wo = None
    
    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        numfeatures = Xtrain.shape[1]
        epochs = self.params["epochs"]

        self.wi = np.random.randn(self.params['nh'], numfeatures)
        self.wo1 = np.random.randn(2, self.params['nh'])
        self.wo2 = np.random.randn(1, self.params['nh'])

        for epoch in range(epochs):
            for i in range(numsamples):
                xtrain = np.reshape(Xtrain[i],(-1,1))
                # Hidden Layer 1
                h1 = self.transfer(np.dot(self.wi,xtrain))
                h1 = np.reshape(h1,(-1,1))
                hout = self.transfer(np.dot(self.wo1,h1))
                
                # Hidden Layer 2
                h2 = self.transfer(np.dot(self.wo1,hout))
                h2 = np.reshape(h2,(-1,1))
                predicted = self.transfer(np.dot(self.wo2,h2))

                # Calculate the gradients
                delta = (-ytrain[i]*(1-hout)) + (1-ytrain[i]) * hout
                grad_wo1 = np.dot(delta,h1.T)
                delta2 = (-hout[i]*(1-predicted)) + (1-hout[i]) * predicted
                grad_wo2 = np.dot(delta2,h2.T)
                grad_int = (self.wo1*h1.T*(1-h1.T)).T
                xtrain = np.reshape(Xtrain[i].T,(-1,1))
                grad_wi = delta*np.dot(grad_int,xtrain.T)

                # Call update function to update the weights
                self.update(grad_wi, grad_wo1, grad_wo2)

    def predict(self,Xtest):
        predictions = []
        numtest = Xtest.shape[0]
        
        for i in range(0,numtest):
            # Retrieve classes label from (ah, ao)
            ao = self.evaluate(Xtest[i])[1]
            if ao > 0.5:
               predictions.insert(i, 1)
            else:
               predictions.insert(i, 0)
               
        return np.reshape(predictions, [numtest, 1])

    def evaluate(self, inputs):
        # hidden activations
        ah = self.transfer(np.dot(self.wi,inputs.T))

        # output activations
        ao = self.transfer(np.dot(self.wo,ah)).T

        return (
            ah, # shape: [nh, samples]
            ao, # shape: [classes, nh]
        )

    def update(self, i, o1, o2):
        self.wi = self.wi - self.params['stepsize'] * i
        self.wo1 =  self.wo1 - self.params['stepsize'] * o1
        self.wo2 =  self.wo2 - self.params['stepsize'] * o2