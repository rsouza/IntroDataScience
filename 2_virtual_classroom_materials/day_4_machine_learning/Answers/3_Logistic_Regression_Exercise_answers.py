# Databricks notebook source
# MAGIC %md
# MAGIC # Logistic regression

# COMMAND ----------

# Importing all the necessary libraries
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics # to calculate accuracy measure and confusion matrix
import matplotlib.pyplot as plt 
import random
plt.rcParams["figure.figsize"] = (15,6)

# COMMAND ----------

# MAGIC %md
# MAGIC # Binary regression

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load dataset for binary regression

# COMMAND ----------

X, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
print(datasets.load_breast_cancer().DESCR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make the data imbalanced
# MAGIC 
# MAGIC For the purpose of this exercise we will make the data imbalanced by removing 80% of the cases where y==1

# COMMAND ----------

data = pd.concat([X,y], axis=1) # join X and y
data_neg = data.loc[data.target==0,:] # select only rows with negative target 
data_pos = data.loc[data.target==1,:].sample(frac=0.07, random_state=42) # select 7% of rows with positive target

data_imb = pd.concat([data_neg, data_pos]) # concatenate 7% of positive cases and all negative ones to have imbalanced data
X_imb = data_imb.drop(columns=['target'])
y_imb = data_imb.target
plt.title('frequency of the target variable')
plt.xlabel('target value')
plt.ylabel('count')
plt.hist(y_imb);


# COMMAND ----------

# MAGIC %md
# MAGIC split to train test

# COMMAND ----------

#Task:

X_train , X_test , y_train , y_test = train_test_split(X_imb, y_imb, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise
# MAGIC 
# MAGIC - fit the default LogisticRegression() to X_train, y_train

# COMMAND ----------

#Task:

lr = LogisticRegression()
lr.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC The model failed to converge due to low number of iterations of the optimization solver. There are multiple solvers that can be chosen as a hyperparameter of the model. These also depend on the strategy that is chosen for regularization and for multiclass problem. Description of which solver suits which problem is in the documentation. We have 3 options now. 
# MAGIC 
# MAGIC - increase number of iterations until the default solver converges
# MAGIC - select a different optimization algorithm with a hyperparameter solver
# MAGIC - scale input data which usually helps optimization algorithms to converge. However, if you do not use regularization, the scaling is not required for a logistic regression. It only helps with a convergence 
# MAGIC 
# MAGIC ### Exercise
# MAGIC 
# MAGIC - scale the data with a StandardScaler()
# MAGIC - fit and transform X_train and save to *X_train_scaled*
# MAGIC - transform X_test and save to *X_test_scaled*

# COMMAND ----------

#Task:

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise
# MAGIC 
# MAGIC - fit the logistic regression to the scaled data
# MAGIC - predict on X_train_scaled and save the values to *y_hat*
# MAGIC - what are the values that are returned from the predict() method?

# COMMAND ----------

#Task:

lr.fit(X_train_scaled, y_train)
y_hat = lr.predict(X_train_scaled)
y_hat

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise
# MAGIC 
# MAGIC - print different metrics from sklearn.metrics for the predictions on the train set
# MAGIC     - accuracy
# MAGIC     - confusion matrix
# MAGIC     - classification report

# COMMAND ----------

#Task:

print(f'accuracy {metrics.accuracy_score(y_train, y_hat)}')
print(f'confusion matrix\n {metrics.confusion_matrix(y_train, y_hat)}')
print(f'classification report\n {metrics.classification_report(y_train, y_hat)}')

# COMMAND ----------

# MAGIC %md
# MAGIC __WARNING__: You should never optimize for the results of the test set. Test set should be always set aside and you should evaluate only once you have decided for the final model. You will learn later in the course how to treat such situations in the lecture about hyperparameter tuning.
# MAGIC 
# MAGIC You can see from the confusion matrix that there are only 19 cases of the positive class in the train set while 2 of them were classified incorrectly and 17 correctly. We would rather want to predict correctly all those cases where target = 1. It is not a big deal if we tell the patient that she/he has a cancer while actually there is no cancer. The bigger problem is if we predict that the patient does not have a cancer while she/he actually has it. We can achieve it by changing the value of the threshold that is by default 50%. We should therefore lower the threshold for the probability.
# MAGIC 
# MAGIC After calling .predict() on your model it returned predicted classes. Instead of predicting classes directly you can return probabilites for each instance using predict_proba() method of logistic regression model. One row is one observation. The first column is the probability that the instance belongs to the first class and the second column tells you about the probability of the instance belonging to the second class. Sum of the first and second column for each instance is equal to 1. Which class is the first and which is the second? You can find out with classes_ attribute of the model.
# MAGIC 
# MAGIC ### Exercise
# MAGIC 
# MAGIC - return classes with classes_ attribute
# MAGIC - return probabilites of the X_train_scaled with a predict_proba() method
# MAGIC - save the probabilities of the positive class into a variable *probs_train*

# COMMAND ----------

#Task:

print(lr.classes_)
print(lr.predict_proba(X_train_scaled))
probs_train = lr.predict_proba(X_train_scaled)[:,1]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise 
# MAGIC 
# MAGIC - define the value of a threshold equal to 20%
# MAGIC - use probabilities saved in the variable *probs_train*. If the value of the probability is >= than threshold then the prediction should be equal to 1. Hint: boolean values can be converted to 0/1 with boolean_values.astype(int)
# MAGIC - return confusion_matrix as well as classification_report for a train set

# COMMAND ----------

#Task:

threshold = 0.2
preds_train = (probs_train>=threshold).astype(int)
print(metrics.confusion_matrix(y_train, preds_train))
print(metrics.classification_report(y_train, preds_train))

# COMMAND ----------

# MAGIC %md
# MAGIC It seems now that all the positive cases are classified correctly thanks to the change of the prediction threshold. Let's check the performance on the test data.
# MAGIC 
# MAGIC ### Exercise
# MAGIC 
# MAGIC - return probabilites of a positive class from the model on the X_test_scaled dataset
# MAGIC - convert the probabilities into predictions with a threshold 20% as above
# MAGIC - return confusion_matrix and a classification_report

# COMMAND ----------

#Task:

probs_test = lr.predict_proba(X_test_scaled)[:,1]
preds_test=(probs_test>=0.2).astype(int)
print(metrics.confusion_matrix(y_test, preds_test))
print(metrics.classification_report(y_test, preds_test))

# COMMAND ----------

# MAGIC %md
# MAGIC Great. The model classifies all the 6 positive cases correctly on a test set. There are 2 cases when the patient did not have a cancer but the model predicted a cancer. What we actually wanted to optimize here is a recall for a positive class as we want to catch as many positive cases as possible. You can see the values of recall for class 1 as a function of a threshold on the chart below

# COMMAND ----------

recalls = []
for threshold in np.linspace(0,1,100):
    preds_train = (probs_train>=threshold).astype(int)
    recalls.append(metrics.classification_report(y_train, preds_train, output_dict=True,zero_division=1)['1']['recall'])
plt.xlabel('threshold')
plt.ylabel('recall for class 1')
plt.title("A search for optimal threshold")
plt.plot(np.linspace(0,1,100), recalls)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC You can return parameters of the fitted model. This is convenient for automatic retraining of the model where you can extract the parameters of the best model and also set the parameters of the model with set_params(**params).

# COMMAND ----------

lr.get_params()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regularization
# MAGIC 
# MAGIC Similarly to linear regression you can apply any of the l1, l2 and elastic net regularization techniques. The strength of the regularization is here defined with a parameter C which is an inverse of alpha. This means that smaller the C the stronger the regularization. Default value is 1.
# MAGIC 
# MAGIC Different regularization techniques work only with certain solver, e.g. for L1 penalty we have to use either liblinear or saga solver, L2 can be handled with newton-cg, lbfgs and sag solvers, elasticnet works only with saga solver. For elasticnet you can adjust parameter l1_ratio.
# MAGIC 
# MAGIC ### Exercise
# MAGIC 
# MAGIC - fit the logistic regression on X_train_scaled with a regularization of your choice with a parameter penalty
# MAGIC - change the solver if needed, see documentation
# MAGIC - try different values of C to see the effect on results, try also stroner values like 0.1, 0.01,...
# MAGIC - predict on X_test_scaled and return classification report

# COMMAND ----------

#Task:

lr = LogisticRegression(penalty='l1', C = 0.1, solver='liblinear')
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
print(metrics.classification_report(y_test, y_pred))

# COMMAND ----------

print(f'coefficients of the logistic regression:\n {lr.coef_}')

# COMMAND ----------

# MAGIC %md
# MAGIC If you fitted for example LogisticRegression(penalty='l1', C = 0.1, solver='liblinear') you would see that many of the coefficients are equal to 0. This behavior of l1 is expected not only for linear but also for logistic regression.

# COMMAND ----------

# MAGIC %md
# MAGIC # Multinomial Logistic Regression

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data
# MAGIC 
# MAGIC We will use here a dataset with a handwritten numbers in a low resolution of 8x8 pixels. One row is 64 values of pixels. There are 10 classes. You can see few examples of obserations in the picture below. We perform also a usual train test split and a scaling of features to help optimizers converge

# COMMAND ----------

data = datasets.load_digits()
X, y = data.data, data.target
X_train , X_test , y_train , y_test = train_test_split(X, y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for i in range(10):
    plt.subplot(2,5,i+1)
    num = random.randint(0, len(data))
    plt.imshow(data.images[num], cmap=plt.cm.gray, vmax=16, interpolation='nearest')

# COMMAND ----------

print(data.DESCR)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise
# MAGIC 
# MAGIC - fit a default logistic regression on X_train_scaled, y_train
# MAGIC - predict and print the classification report on X_test_scaled

# COMMAND ----------

#Task:

lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_hat = lr.predict(X_test_scaled)

print(metrics.classification_report(y_test, y_hat)) # zero_division=1

# COMMAND ----------

# MAGIC %md
# MAGIC You can see that in the classification report there is 1 row per 1 class with all the statistics.
# MAGIC 
# MAGIC If you return probabilites with the predict_proba() method you will see that it has 1 column per 1 class. It is a generalization of the binary case. The sum of all the probabilities per 1 row is equal to 1

# COMMAND ----------

probs = lr.predict_proba(X_test_scaled)
print(f'predict_proba shape: {probs.shape}')

# COMMAND ----------

# MAGIC %md
# MAGIC Logistic regression can handle multinomial regression without any special setting. There is however a parameter that lets you choose the strategy for the multinomial problem. It is either one_vs_rest or softmax regression. The choice of the strategy is also dependent on the selected solver. I.e. if the solver = 'liblinear' then a softmax regression is not possible. In this case and if the problem is binary, the default strategy for multi_class is one vs rest. Otherwise it is softmax
# MAGIC 
# MAGIC ### Exercise
# MAGIC - fit logistic regression on X_train_scaled, y_train. use parameter multi_class with a value 'ovr' which is one versus rest strategy
# MAGIC - return probabilities

# COMMAND ----------

#Task:

lr = LogisticRegression(multi_class='ovr')
lr.fit(X_train_scaled, y_train)
y_hat = lr.predict(X_test_scaled)
probs = lr.predict_proba(X_test_scaled)
print(f'predict_proba shape: {probs.shape}')
np.sum(probs,axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ------------------------------------------------------------------------------------------------------------
# MAGIC Some material adapted for RBI internal purposes with full permissions from original authors. [Source](https://github.com/zatkopatrik/authentic-data-science)
