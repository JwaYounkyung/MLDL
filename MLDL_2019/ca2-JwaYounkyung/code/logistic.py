"""
A starting code for a logistic regression model.  This implementation should be based on the
minimum classification error heuristic.
"""

from numpy import *

from binary import *
import util
from regression import *

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from scipy.special import expit

def sigmoid(z):
    return 1. / (1 + exp(-z))

def softmax(z):
    """Softmax function"""
    if sum(exp(-z)) == 0 :
        return zeros(z.shape[0])
    return exp(-z) / sum(exp(-z))

class Logistic(BinaryClassifier):
    """
    This class is for the logistic regression model implementation.
    """

    w = None
    gamma = 0.0

    def __init__(self):
        """
        Initialize our internal state.
        """

    def setLambda(self, lambdaVal):
        self.gamma = lambdaVal

    def online(self):
        ### TODO: YOU MAY MODIFY THIS
        return False

    def __repr__(self):
        """
        Return a string representation of the model
        """
        return self.w

    def __str__(self):
        """
        Return a string representation of the model
        """
        return self.w

    def predict(self, X):
        """
        Perform inference
        """
        ### TODO: YOUR CODE HERE

        X_m = array(X, dtype=float)

        return matmul(X_m, self.w)

        #util.raiseNotDefined()

    def train(self, X, Y):
        """
        Build a logistic regression model.
        """
        ### TODO: YOUR CODE HERE
        X_m = array(X, dtype=float)
        Y_m = array(Y, dtype=float)

        w = zeros(X_m.shape[1])
        #self.w = zeros((X_m.shape[1],1))
        self.w = zeros(X_m.shape[1])

        lr = 0.01 #learning rate
        n_epochs = 100 #num of epochs
        ii=0
        #for epoch in range(n_epochs):
        while True:
            ii = ii +1
            loss1 = mean(square(Y_m - matmul(X_m, self.w)))
            for i in range(X_m.shape[1]) :
                t_X_m = transpose(X_m[:,i:i+1])
                xw = matmul(X_m, self.w)
                soft = softmax(xw)
                sum_ys = sum(Y_m - soft)
                w[i] = self.w[i] + lr*sum_ys
                #sum_xt = matmul(t_X_m, (Y_m - soft))
                #w[i] = self.w[i] + lr*sum_xt[0]
                
            self.w = w
            print(self.w)
            loss2 = mean(square(Y_m - matmul(X_m, self.w)))

            if ii != 1 and ii !=2 and loss1 < loss2 :
                break


            #print("comp" + str(epoch))
            #cost = mean(square(subtract(matmul(X_m,self.w), Y_m)))
            #if epoch % 20 == 0:
                #print("w: " + str(self.w)) 

        #util.raiseNotDefined()

    def traintestskit(self, X, Y, X_t, Y_t):
        Y_m = array(Y, dtype=float)#.reshape(-1,1)
        Y_t_m = array(Y_t, dtype=float)#.reshape(-1,1)
        
        X_t = array(X_t, dtype=float)

        H = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')
        H.fit(X, Y_m)
        Y_pred = H.predict(X_t)
        
        print(H.coef_)
        print("Mean squared error: %f" % mean_squared_error(Y_t_m, Y_pred))


