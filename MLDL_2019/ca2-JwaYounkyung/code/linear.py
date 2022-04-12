"""
A starting code for a vanilla linear regression model.  This implementation should be based on the
minimum classification error heuristic.
"""

from numpy import *
from numpy.linalg import inv

from binary import *
import util
from regression import *

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

class Linear(Regression):
    """
    This class is for the decision tree implementation.  //wrong
    It has a partial implementation for the tree data structure. 
    This class also has a function to print the tree in a canonical form.
    """

    w = None

    def __init__(self):
        """
        Initialize our internal state.  The options are:
          opts.maxDepth = maximum number of features to split on
                          (i.e., if maxDepth == 1, then we're a stump) //wrong
        """

    def online(self):
        ### TODO: YOU MAY MODIFY THIS
        return False

    def __repr__(self):
        """
        Return a string representation of the tree //wrong
        """
        return self.w

    def __str__(self):
        """
        Return a string representation of the tree //wrong
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
        Build a linear regressor.
        """
        ### TODO: YOUR CODE HERE

        X_m = array(X, dtype=float)
        Y_m = array(Y, dtype=float)

        X_transe = transpose(X_m)

        w_closed = matmul(inv(matmul(X_transe, X_m)), matmul(X_transe, Y_m))
        
        print(w_closed)

        self.w = w_closed

        #util.raiseNotDefined()

    def traintestskit(self, X, Y, X_t, Y_t):
        Y_m = array(Y, dtype=float)#.reshape(-1,1)
        Y_t_m = array(Y_t, dtype=float)#.reshape(-1,1)

        X_t = array(X_t, dtype=float)

        H = linear_model.LinearRegression()
        H.fit(X, Y_m)
        Y_pred = H.predict(X_t)
        
        print(H.coef_)
        print("Mean squared error: %lf" % mean_squared_error(Y_t_m, Y_pred))
        
