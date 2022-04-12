# Coding Assignment 2: Implementing three linear regression models for estimating housing price given features.

## Introduction

The second coding assignment asks you to implement three linear regression models to predict house price given features.

**Note**: we will use `Python 2.x` for the project. If you want to use `Python 3.x`, you have to modify some of the utility functions (but not much. It should be doable).

We provide the code consisting of several Python files, some of which you will need to read and understand in order to complete the assignment, and some of which you can ignore. You will figure them out which one is which.

---
**Installing prerequisites**

The prerequisite usually refers to the necessary library that your code can run with. They are also known as `dependency`. To install the prerequisite, simply type in the shell prompt (not in a python interpreter) the following:

```
$ pip install -r requirements.txt
```

---

**Files you'll edit:**

`datasets.py` Data provider. Implement read the `./dataset/price_data_tr.csv` and `./dataset/price_data_val.csv` file. The recommended interface has been noted in the file as comments.

`dumbClassifiers.py` contains "warm up" classifiers to get you used to our this code framework.

`vanillaLinear.py` will be your implementation of vanilla linear regression model.

`ridge.py` will be your implementation of ridge regression model.

`logistic.py` will be your implementation of logistic regression model.

---

**Files you may want to review:**

`binary.py` A generic interface for binary classifiers or regressions.

`util.py` Utility functions!

`runClassifier.py` A few wrapper functions for running classifiers: training, generating learning curves, etc.

`mlGraphics.py` Plotting commands.

---

**What to submit (Push to your github classroom):** 
- All of the python files listed above (under "Files you'll edit"). 
- `report.pdf` file that answers all the written questions in this assignment (denoted by `"REPORT#:"` in this documentation).

<!-- **Evaluation:** Your code will be autograded for technical correctness. Please do not change the names of any provided functions or classes within the code, or you will wreak havoc on the autograder. However, the correctness of your implementation -- not the autograder's output -- will be the final judge of your score. If necessary, we will review and grade assignments individually to ensure that you receive due credit for your work. -->

**Academic Dishonesty:** We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else's code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don't try. We trust you all to submit your own work only; please don't let us down. If you do, we will pursue the strongest consequences available to us.

**Getting Help:** You are not alone! If you find yourself stuck on something, contact the course staff for help. Office hours, class time, and Piazza are there for your support; please use them. We want these projects to be rewarding and instructional, not frustrating and demoralizing. But, we don't know when or how to help unless you ask.

<!-- **One more piece of advice:** if you don't know what a variable is, print it out. -->

---
### Prepare the dataset (5%)

<!-- First, unzip the `price_data.csv.zip` file in `./dataset` to obtain `price_data_tr.csv` and `price_data_val.csv`. I have zipped them to reduce time to clone the repo. -->

Read the csv to load the dataset.

```
>>> import datasets
>>> price_dataset = datasets.PriceDataset()
>>> [tr_x, tr_y, val_x, val_y] = price_dataset.getDataset()
```

You may ignore the warning of `Fontconfig warning: ignoring UTF-8: not a valid region tag` after `import datasets` command.

---

### Vanilla Linear Regression Model (10%)

With the experience of implementing other classifiers in CA 1, you can now implement the linear regression model to predict the price (`y`) with other data (`x`).

```
>>> import linear
>>> model = Linear()
>>> model.train(tr_x, tr_y)
>>> y_hat = model.predict(val_x)
>>> error = computeAvgRegrMSError(val_y, y_hat) 
>>> print error
0.241  # for example
```

`REPORT1`: Report the error. Discuss any ideas to reduce the errors (e.g., new feature transforms or using kernels or etc.)

---
### Ridge Regression Model (10%)

Same as before but implement a ridge regression model. 

```
>>> import ridge
>>> import runClassifier
>>> model = Ridge()
>>> lambda = 1.0
>>> model.setLambda(lambda)
>>> model.train(tr_x, tr_y)
>>> y_hat = model.predict(val_x)
>>> error = computeAvgRegrMSError(val_y, y_hat) 
>>> print error
0.1542  # for example
```

`REPORT2`: Sweep `gamma` from 0.0 to 1.0 (or some other reasonable values), plot a graph (x-axis: gamma, y-axis: accuracy) and discuss the effect of the gamma (especially comparing with vanilla linear when `gamma=0`.)


---
### Multi-class Logistic Regression (Optimized by Stochastic Gradient Descent) (25%)

In class, we learn the logistic regression model is for binary classification. Here, we are solving a regression problem (although the name of the classifier seems for regression). To change the regression problem into classification problem, we can quantize the price by 100,000 steps (then you'll have 77 classes). The regression precision would be bounded by 100,000 but you can solve the problem. Or you can encode the steps more fine-grained or varied step sized (eg., In 75,000 to 1,000,000, quantize it by 50K and more than 1M as a single class).

We ask you to implement multi-class Logistic regression model. In class, we learn how to implement binary class logistic regression model. There are two ways of implementing multi-class logistic regression model. 1) a one-versus-all binary logistic regression models, 2) changing logistic loss to cross entropy loss [a useful page](https://peterroelants.github.io/posts/cross-entropy-logistic/).

```
>>> import logistic
>>> import runClassifier
>>> import numpy as np
>>> models = []
>>> tr_y_qtzd = quantizeY(tr_y)
>>> y_classes = np.unique(tr_y_qtzd)
>>> num_classes = len(y_classes)
>>> for i in range(num_classes):
...     model = Logistic()
...     model.train(tr_x, tr_y_qtzd)
...     models.append(model)
>>> max_conf = 0
>>> max_class_idx = 0
>>> for i in range(num_classes):
...     _, confidence = models[i].predict(val_x)
...     if confidence > max_conf:
...         max_conf = confidence
...         max_class_idx = i
>>> val_y_hat = y_classes[max_class_idx]
>>> error = computeAvgRegrMSError(val_y, y_hat) 
>>> print error
0.1387  # for example
```

`REPORT4`: Report the error. Discuss any improvemental ideas (eg., add regularizers.).

`REPORT5`: Discuss any idea to solve the regression problem by converting it classification problem.


---
### Compare your implementations with `scikit-learn` library (20%)

In [scikit-learn library](https://scikit-learn.org/), there are all implementation of 1) [vanilla linear regressor](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares), 2) [ridge regressor](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression) and 3) [logistic regression model](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression). Please make sure your implementations of vanilla linear regressor and ridge regressor are *exactly* the same to the ones in scikit-learn. Please compare outputs of your version of logistic regressions and the scikit-learn version.

`REPORT6`: Compare the error by your implementations of vanilla linear regression and OLS model in scikit-learn and discuss the reason for the difference. If they are identical, report and claim you're awesome!

`REPORT7`: Compare the error by your implementations of ridge regression and ridge regression model in scikit-learn and discuss the reason for the difference. If they are identical, report and claim you're awesome!

`REPORT8`: Compare the error by your implementations of logistic regression and logistic regression model in scikit-learn and discuss the reason for the difference. If they are identical, report and claim you're awesome!

---
### Compete the accuracy with others using Kaggle (30%)

Using either your implementation of other regressors or tune the outputs of libraries implemented in scikit-learn, please get the best accuracy of price prediction and compete with other students in the Kaggle platform. We'll release the competition detailed soon in a Piazza posting.

If you use your implementation to win the competition, we will give you big extra credits.

`REPORT9`: Discuss all trials you've done with either with your implementations or scikit-learn library's functions.
