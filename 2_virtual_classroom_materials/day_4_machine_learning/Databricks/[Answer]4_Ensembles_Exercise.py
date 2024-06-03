# Databricks notebook source
# MAGIC %md
# MAGIC # Ensembles
# MAGIC
# MAGIC We have seen in the slide presentation that ensemble methods are better than simple classifiers such as decision trees. But how do ensembles compare? Which one should we use? The answer is that there is no silver bullet, and it dependes on the data, the task and the parameters.  
# MAGIC Let's explore some ensemble techniques:

# COMMAND ----------

# Importing all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

plt.rcParams["figure.figsize"] = (15, 6)

# COMMAND ----------

# MAGIC %md
# MAGIC The three most popular methods for combining the predictions from different models are:
# MAGIC
# MAGIC + [**Bagging/Pasting**](https://en.wikipedia.org/wiki/Bootstrap_aggregating)    
# MAGIC     Building multiple models (typically of the same type) from different subsamples of the training dataset.  
# MAGIC + [**Boosting**](https://en.wikipedia.org/wiki/Boosting_(machine_learning%29)   
# MAGIC     Building multiple models (typically of the same type) each of which learns to fix the prediction errors of a prior model in the chain.  
# MAGIC + [**Stacking/Voting**](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking)    
# MAGIC     Building multiple models (typically of differing types) and creating a meta-learner with the features of the models, or simple statistics (like calculating the mean) to combine predictions.  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load dataset for ensembles comparison
# MAGIC
# MAGIC We are loading the Pima Indians Diabetes Database .
# MAGIC This data set is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the data set is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the data set. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
# MAGIC
# MAGIC **Content**
# MAGIC
# MAGIC The datasets consists of several medical predictor variables and one target variable, 'Outcome'. Predictor variables include the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
# MAGIC
# MAGIC Acknowledgements  
# MAGIC Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.

# COMMAND ----------

names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
df = pd.read_csv("../../../Data/pima-indians-diabetes.data.csv", names=names)
df.head()

# COMMAND ----------

array = df.values
X = array[:, 0:8]
Y = array[:, 8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed, shuffle=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Note that even if we set a random seed, our results may vary given the stochastic nature of the algorithm or evaluation procedure, or differences in numerical precision. Consider running the example a few times and compare the average outcome.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision Tree
# MAGIC First, let's try with a simple [`DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/tree.html).

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier

cart1 = DecisionTreeClassifier(random_state=seed)
results1 = cross_val_score(cart1, X, Y, cv=kfold)
print(results1.mean())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bagged Decision Trees  
# MAGIC Now, let's use the [Scikit-Learn bagging classifier](https://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator) to manually create a Random Forest:

# COMMAND ----------

from sklearn.ensemble import BaggingClassifier

num_trees = 100
cart2 = DecisionTreeClassifier()
model2 = BaggingClassifier(estimator=cart2, n_estimators=num_trees, random_state=seed)
results2 = cross_val_score(model2, X, Y, cv=kfold)
print(results2.mean())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest
# MAGIC
# MAGIC [Random forest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests-and-other-randomized-tree-ensembles) is an extension of bagged decision trees.
# MAGIC Samples of the training dataset are taken with replacement, but the trees are constructed in a way that reduces the correlation between individual classifiers. Specifically, rather than greedily choosing the best split point in the construction of the tree, only a random subset of features are considered for each split.

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

max_features = "sqrt"
model3 = RandomForestClassifier(
    n_estimators=num_trees, 
    max_features=max_features, 
    random_state=seed
)
results3 = cross_val_score(model3, X, Y, cv=kfold)
print(results3.mean())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extra Trees
# MAGIC
# MAGIC [Extra Trees](https://quantdare.com/what-is-the-difference-between-extra-trees-and-random-forest/)
# MAGIC are another modification of bagging where random forests are constructed from the whole training dataset.
# MAGIC You can construct an Extra Trees model for classification using the
# MAGIC [`ExtraTreesClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
# MAGIC of
# MAGIC [`sklearn.ensemble`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)
# MAGIC class.

# COMMAND ----------

from sklearn.ensemble import ExtraTreesClassifier

max_features = 7
model4 = ExtraTreesClassifier(
    n_estimators=num_trees, max_features=max_features, random_state=seed
)
results4 = cross_val_score(model4, X, Y, cv=kfold)
print(results4.mean())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Boosting Algorithms  
# MAGIC
# MAGIC
# MAGIC Boosting ensemble algorithms create a sequence of models that attempt to correct the mistakes of the models before them in the sequence. 
# MAGIC Once created, the models make predictions which may be weighted by their demonstrated accuracy and the results are combined to create a final output prediction. The two most common boosting ensemble machine learning algorithms are:
# MAGIC
# MAGIC + AdaBoost
# MAGIC + Stochastic Gradient Boosting

# COMMAND ----------

# MAGIC %md
# MAGIC ### AdaBoost
# MAGIC
# MAGIC [AdaBoost](https://scikit-learn.org/stable/modules/ensemble.html#adaboost) was perhaps the first successful boosting ensemble algorithm. It generally works by weighting instances in the dataset by how easy or difficult they are to classify, allowing the algorithm to pay or or less attention to them in the construction of subsequent models.  
# MAGIC You can construct an AdaBoost model for classification using the AdaBoostClassifier class.  

# COMMAND ----------

from sklearn.ensemble import AdaBoostClassifier

model5 = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results5 = cross_val_score(model5, X, Y, cv=kfold)
print(results5.mean())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stochastic Gradient Boosting  
# MAGIC
# MAGIC [Stochastic Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting) (also called Gradient Boosting Machines) is one of the most sophisticated ensemble techniques. It is also a technique that is currently proving to be perhaps of the the best techniques available for improving performance via ensembles.  
# MAGIC You can construct a Gradient Boosting model for classification using the GradientBoostingClassifier class.  

# COMMAND ----------

from sklearn.ensemble import GradientBoostingClassifier

model6 = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results6 = cross_val_score(model6, X, Y, cv=kfold)
print(results6.mean())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Voting Ensemble
# MAGIC
# MAGIC [Voting](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier) is one of the simplest ways of combining the predictions from multiple machine learning algorithms.
# MAGIC It works by first creating two or more standalone models from your training dataset. A Voting Classifier can then be used to wrap your models and average the predictions of the sub-models when asked to make predictions for new data.
# MAGIC
# MAGIC The predictions of the sub-models can be weighted, but specifying the weights for classifiers manually or even heuristically is difficult. 
# MAGIC You can create a voting ensemble model for classification using the
# MAGIC [`VotingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html) class.
# MAGIC The code below provides an example of combining the predictions of logistic regression, classification and regression trees and support vector machines for a classification problem.

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

estimators1 = [
    ("logistic", LogisticRegression(solver="lbfgs", max_iter=300, random_state=seed)),
    ("cart", DecisionTreeClassifier(random_state=seed)),
    ("svm", SVC(random_state=seed)),
]

# Create the ensemble model
ensemble1 = VotingClassifier(estimators1)
results7 = cross_val_score(ensemble1, X, Y, cv=kfold)
print(results7.mean())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stacking Classifier
# MAGIC
# MAGIC More advanced methods, similar to voting, can learn how to best weight the predictions from submodels ([stacked generalization](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization)).   
# MAGIC We can use the
# MAGIC [`StackingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html) from Scikit-Learn.

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier

estimators2 = [
    (
        "logistic",
        make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="lbfgs", max_iter=300, random_state=seed),
        ),
    ),
    ("rf", RandomForestClassifier(n_estimators=num_trees, random_state=seed)),
    ("svm", make_pipeline(StandardScaler(), SVC(random_state=seed))),
]


ensemble2 = StackingClassifier(
    estimators=estimators2,
    final_estimator=LogisticRegression(solver="lbfgs", max_iter=300, random_state=seed),
)
results8 = cross_val_score(ensemble2, X, Y, cv=kfold)
print(results8.mean())

# COMMAND ----------

# MAGIC %md
# MAGIC You can check how we have improved simple classifiers using ensembles.

# COMMAND ----------

# MAGIC %md
# MAGIC # Task
# MAGIC
# MAGIC + Change the parameters of the estimators above.
# MAGIC + Try to optmize and discuss the results.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample solution
# MAGIC
# MAGIC There is no single one correct way of solving this task.
# MAGIC It is meant to get you working with Ensemble estimators.
# MAGIC The code below is meant to inspire you to come up with your own insights.
# MAGIC
# MAGIC First we want to get an idea what we are comparing our classifiers with. 
# MAGIC To this end we fit a [`DummyClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html).
# MAGIC By default the dummy classifier always predicts the majority class in the training set.
# MAGIC In our task this is `class == 0` which is about 65% of the observations and thus the mean accuracy for the cross vaildation is about the same.

# COMMAND ----------

from sklearn.dummy import DummyClassifier

dummy = DummyClassifier()
task1 = cross_val_score(dummy, X, Y, cv=kfold)
print(task1.mean())

# COMMAND ----------

# MAGIC %md
# MAGIC Considering the above the performance of the [`DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/tree.html) is not very good.
# MAGIC
# MAGIC The random Forrest bagging classifiers all improve the result by about 10 percent points each.
# MAGIC A similar result is achieved when we utilize the Bagging classifiers with default parameters.
# MAGIC
# MAGIC At the end we fit two ensemble classifiers both of them improving the baseline result by 12 percent points.
# MAGIC However non of the classifiers were properly tuned.
# MAGIC An extensive introduction to Hyperparameter tuning will be given in later modules of this course.
# MAGIC
# MAGIC We will try to improve the [`DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/tree.html) by trying different values for the `max_depth` parameter.

# COMMAND ----------

for i in range(1, 10):
    decision_tree = DecisionTreeClassifier(max_depth=i)
    task2 = cross_val_score(decision_tree, X, Y, cv=kfold)
    print(f"Decision Tree, max_depth={i} result: Mean Accuracy = {task2.mean()}")

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see above, if we limit the depth of the tree the result can be improved to match the default random forrest classifieres of about 75%. 
# MAGIC It it worth keeping in mind that for some problems it is actually usefull to spend some time on tuning simpler models befor using more complex ones.
# MAGIC
# MAGIC Of course we can also try to improve one of ensemble classifiers.
# MAGIC Again we don't expect you to use proper hyperparameter tuning at this point in the course.
# MAGIC If you are interested in the topic have a look at [grid search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search) which is one of the fundamental techniques.
# MAGIC
# MAGIC Apart from optimizing our hyperparameters we can also experiment with adding more classifiers to our voting ensemble.

# COMMAND ----------

from sklearn.naive_bayes import *

estimators3 = [
    ("logistic", LogisticRegression(solver="lbfgs", max_iter=300, random_state=seed)),
    ("tree", DecisionTreeClassifier(random_state=seed)),
    ("svm", SVC(random_state=seed)),
    ("rf", RandomForestClassifier(n_estimators=num_trees, random_state=seed)),
    ("ada", AdaBoostClassifier(n_estimators=num_trees, random_state=seed)),
    ("nb", GaussianNB()),
]

# Create the ensemble model
voting = VotingClassifier(estimators3)
task3 = cross_val_score(voting, X, Y, cv=kfold)
print(task3.mean())

# COMMAND ----------

# MAGIC %md
# MAGIC As always more does not automatically mean better.
# MAGIC Most of the time simply adding more classifiers will not improve the result.
# MAGIC
# MAGIC All of the classifiers themself need to perform well to leverage the benefits of voting.

# COMMAND ----------

from sklearn.naive_bayes import *

estimators3 = [
    ("svm", SVC(random_state=seed)),
    ("rf", RandomForestClassifier(n_estimators=num_trees, random_state=seed)),
    ("nb", GaussianNB()),
]

# Create the ensemble model
voting = VotingClassifier(estimators3)
task3 = cross_val_score(voting, X, Y, cv=kfold)
print(task3.mean())

# COMMAND ----------

# MAGIC %md
# MAGIC Additional reading: [scaling and adjust LR models](https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati)  
# MAGIC Some material inspired by this [Source](https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/)
