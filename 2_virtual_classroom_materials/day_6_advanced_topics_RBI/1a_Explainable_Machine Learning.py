# Databricks notebook source
# MAGIC %md
# MAGIC ## Partial Dependence and Individual Conditional Expectation plots
# MAGIC
# MAGIC Partial dependence plots (PDP) and individual conditional expectation (ICE) plots can be used to visualize and analyze interaction between the target response [1] and a set of input features of interest.
# MAGIC
# MAGIC Both PDPs [H2009] and ICEs [G2015] assume that the input features of interest are independent from the complement features, and this assumption is often violated in practice. Thus, in the case of correlated features, we will create absurd data points to compute the PDP/ICE [M2019].

# COMMAND ----------

# MAGIC %md
# MAGIC ### [Partial Dependence Plots](https://scikit-learn.org/stable/modules/partial_dependence.html)
# MAGIC
# MAGIC Partial dependence plots (PDP) show the dependence between the target response and a set of input features of interest, marginalizing over the values of all other input features (the ‘complement’ features). Intuitively, we can interpret the partial dependence as the expected target response as a function of the input features of interest.
# MAGIC
# MAGIC Due to the limits of human perception, the size of the set of input features of interest must be small (usually, one or two) thus the input features of interest are usually chosen among the most important features.

# COMMAND ----------

 !pip install -U -q scikit-learn

# COMMAND ----------

import matplotlib.pyplot as plt

from sklearn.datasets import make_hastie_10_2
from sklearn.datasets import load_iris

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay

plt.rcParams['figure.figsize'] = [14, 10]

# COMMAND ----------

X, y = make_hastie_10_2(random_state=0)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf.fit(X, y)

# COMMAND ----------

features = [0, 1, (0, 1)]
result = PartialDependenceDisplay.from_estimator(clf, X, features)

# COMMAND ----------


iris = load_iris()
mc_clf = GradientBoostingClassifier(n_estimators=10, max_depth=1)
mc_clf.fit(iris.data, iris.target)

# COMMAND ----------

features = [3, 2, (3, 2)]
PartialDependenceDisplay.from_estimator(mc_clf, X, features, target=0)
