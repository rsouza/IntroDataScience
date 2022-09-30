# Databricks notebook source
# MAGIC %md
# MAGIC # Regularized Linear Models
# MAGIC 
# MAGIC [Good article on regularization for ML](https://towardsdatascience.com/understanding-regularization-in-machine-learning-d7dd0729dde5)

# COMMAND ----------

from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,15)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data
# MAGIC 
# MAGIC We load boston data from sklearn.datasets and split to train and test data. Additionally, we generate polynomial features of the second degree. We will work further with x_train_poly, y_train, x_test_poly and y_test. Run the cell below.

# COMMAND ----------

# downloaded original from  "http://lib.stat.cmu.edu/datasets/boston"
# dataset will be deprecated on sklearn datasets library and we have downloaded it here
# The line below was the old way to load the dataset

#data = datasets.load_boston()  

raw_df = pd.read_csv('../Data/Boston.csv')

y = pd.DataFrame(raw_df['target'])
x = pd.DataFrame(raw_df.iloc[:,1:-1])

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)

poly = PolynomialFeatures(2)
x_train_poly = poly.fit_transform(X_train)
x_test_poly = poly.transform(X_test)
poly_names = poly.get_feature_names()

names_dict = {'x0': X_train.columns[0],
             'x1': X_train.columns[1],
             'x2': X_train.columns[2],
             'x3': X_train.columns[3],
             'x4': X_train.columns[4],
             'x5': X_train.columns[5],
             'x6': X_train.columns[6],
             'x7': X_train.columns[7],
             'x8': X_train.columns[8],
             'x9': X_train.columns[9],
             'x10': X_train.columns[10],
             'x11': X_train.columns[11],
             'x12': X_train.columns[12]
            }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise
# MAGIC 
# MAGIC - how many features are there in total?

# COMMAND ----------

# Task

x_train_poly.shape[1]

# COMMAND ----------

# MAGIC %md
# MAGIC We will further use the user defined function plot_coef that takes as input coefficients as output of the fitted model. It plots the coefficient values and calculates average. Run the cell below

# COMMAND ----------

def plot_coef(lr_coef):
    '''
    The function plots coefficients' values from the linear model.
    --------
    params:
        lr_coef: coefficients as they are returned from the classifier's attributes
    '''
    lr_coef = lr_coef.reshape(-1,1)
    print(f'AVG coef value: {np.mean(lr_coef)}')
    plt.plot(lr_coef)
    plt.title("Coefficients' values")
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit linear regression without regularization
# MAGIC 
# MAGIC ### Exercise
# MAGIC 
# MAGIC - instantiate linear regression under the name lr
# MAGIC - fit lr to x_train_poly, y_train 
# MAGIC - predict with lr on x_train_poly and store the results to y_hat_train
# MAGIC - predict with lr on x_test_poly and store the results to y_hat_test
# MAGIC - return RMSE for y_hat_train as well as for y_hat_test. 
# MAGIC 
# MAGIC How do you interpret the difference in performance of the model on train and on test dataset? Can you tell if the model overfits/underfits?

# COMMAND ----------

# Task

lr = LinearRegression()
lr.fit(x_train_poly, y_train)

y_hat_train = lr.predict(x_train_poly)
y_hat_test = lr.predict(x_test_poly)

print(f"RMSE train: {mean_squared_error(y_train, y_hat_train, squared=False)}")
print(f"RMSE test: {mean_squared_error(y_test, y_hat_test, squared=False)}")

# COMMAND ----------

# MAGIC %md
# MAGIC RMSE is almost twice as big for a test set than for a train set. It suggests an overfitting and a poor generalization power of the model.
# MAGIC 
# MAGIC 
# MAGIC - use function plot_coef on coefficients of the fitted model to see the values of the coefficients and the average value of the coefficients.

# COMMAND ----------

plot_coef(lr.coef_)

# COMMAND ----------

# MAGIC %md
# MAGIC The coefficients in a combination with the error values on train and test suggest that we deal here with overfitting of the model on the given set of polynomial features. We should therefore use regularization. 
# MAGIC 
# MAGIC ## Standardization
# MAGIC 
# MAGIC Before fitting any regularized model scaling of features is crucial. Otherwise the regularization would not be fair to features of different scales. Regularized linear models assume that the inputs to the model have zero mean and a variance in the same magnitude. StandarScaler() deducts mean and divides by a standard deviation. 
# MAGIC 
# MAGIC ### Exercise
# MAGIC 
# MAGIC - instantiate StandardScaler() under the name scaler
# MAGIC - fit_transform x_train_poly with a scaler and store the result into x_train_scaled
# MAGIC - once the scaler is fit to x_train_poly you can directly transform x_test_poly with the scaler and store into variable X_test_scaled. You never want to fit on a test sample not to leak information from test data. Test data serves only for evaluation.

# COMMAND ----------

# Task

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train_poly)
X_test_scaled = scaler.transform(x_test_poly)

# COMMAND ----------

# MAGIC %md
# MAGIC If you applied standardization correctly you should see on the bottom chart distributions of all the features concentrated around zero with similar ranges of deviation

# COMMAND ----------

plt.figure(figsize=(10,12))
plt.subplot(2,1,1)
plt.title('Original polynomial features')
plt.boxplot(x_train_poly)

plt.subplot(2,1,2)
plt.title('Scaled features')
plt.boxplot(X_train_scaled)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Lasso
# MAGIC documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
# MAGIC 
# MAGIC ### Exercise
# MAGIC - instantiate Lasso regression under the name lr_l
# MAGIC - fit the model to X_train_scaled, y_train
# MAGIC - predict on X_train_scaled and X_test_scaled and store the predictions to y_hat_train and y_hat_test respectively
# MAGIC - Did the overfit change?

# COMMAND ----------

# Task


from sklearn.linear_model import Lasso

lr_l = Lasso()
lr_l.fit(X_train_scaled, y_train)

y_hat_train = lr_l.predict(X_train_scaled)
y_hat_test = lr_l.predict(X_test_scaled)

print(f"RMSE train: {mean_squared_error(y_train, y_hat_train, squared=False)}")
print(f"RMSE test: {mean_squared_error(y_test, y_hat_test, squared=False)}")

# COMMAND ----------

# MAGIC %md
# MAGIC The performance seems to be comparable on train and test dataset. Hence, the model's generalization power is good now.
# MAGIC 
# MAGIC ### Exercise
# MAGIC 
# MAGIC - use plot_coef() on the coefficients of the lasso model

# COMMAND ----------

# Task

plot_coef(lr_l.coef_)

# COMMAND ----------

# MAGIC %md
# MAGIC The average value of the coefficients is now much smaller. Also many of the coefficients are equal to 0.

# COMMAND ----------

print(f'After applying Lasso on polynomial scaled features we remain with {np.sum(lr_l.coef_!=0)} variables')
print('\nThe selected variables are:\n')
[print(val) for val in pd.DataFrame(poly_names)[lr_l.coef_!=0].values];
print('\nmapping from polynomial names to original feature names: ')
display(names_dict)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise
# MAGIC 
# MAGIC - subset from X_train_scaled only those variables that have a non-zero coefficient and store the subset in the variable *x_train_lasso*
# MAGIC - do the same selection on X_test_scaled and save to *x_test_lasso*
# MAGIC - how many variables are remaining? Check it with the cell above

# COMMAND ----------

# Task

x_train_lasso = X_train_scaled[:,lr_l.coef_!=0]
x_test_lasso = X_test_scaled[:,lr_l.coef_!=0]
x_test_lasso.shape[1]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ridge

# COMMAND ----------

# MAGIC %md
# MAGIC https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
# MAGIC 
# MAGIC We have effectively performed a feature selection with Lasso. Now we will build on it and will use only the selected features in x_train_lasso, x_test_lasso. 
# MAGIC 
# MAGIC Let's try different values for the strength of the optimization, alpha. By default it is equal to 1 and it must be a positive value. Larger values specify stronger regularization. Alpha can be set also in Lasso and Elastic Net.
# MAGIC 
# MAGIC ### Exercise
# MAGIC - fit the ridge regression to x_train_lasso, y_train with the values of alpha 0.001, 0.01, 0.1, 1, 10, 100 to see the effect of the regularization strength.
# MAGIC - return RMSE for x_train_lasso for each of the alpha options
# MAGIC - select a parameter alpha for which the model has the best RMSE 

# COMMAND ----------

# Task

rmses = []
alphas = [0.001, 0.01, 0.1, 1, 10, 100]

for alpha in alphas:    
    lr_r = Ridge(alpha=alpha)
    lr_r.fit(x_train_lasso, y_train)
    y_hat_train = lr_r.predict(x_train_lasso)
    rmses.append(mean_squared_error(y_train, y_hat_train, squared=False))

plt.figure(figsize=(10,12))
plt.title('Errors as a function of a regularization strength')
plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.plot(alphas, rmses);
print(f'The lowest RMSE on a train set is {np.round(np.min(rmses),2)} with the alpha = {alphas[np.argmin(rmses)]}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise
# MAGIC - fit the model with the best performance on train data
# MAGIC - calculate RMSE on the x_test_lasso for the best model

# COMMAND ----------

# Task

lr_r_best = Ridge(alpha=alphas[np.argmin(rmses)]).fit(x_train_lasso, y_train)
y_hat_test = lr_r_best.predict(x_test_lasso)
rmse_test = np.round(mean_squared_error(y_test, y_hat_test, squared=False))
print(f"RMSE test: {np.round(rmse_test,2)}")

# COMMAND ----------

# MAGIC %md
# MAGIC RMSES on the train and the test set are similar.
# MAGIC 
# MAGIC ### Exercise
# MAGIC - use the function plot_coef on the coefficients from the best model to see the coefficients values with their average

# COMMAND ----------

# Task

plot_coef(lr_r_best.coef_)

# COMMAND ----------

# MAGIC %md
# MAGIC # Elastic Net

# COMMAND ----------

# MAGIC %md
# MAGIC Elastic Net is a combination of Lasso and Ridge which is defined by a parameter l1_ratio. If it is equal to 1 the model is equivalent to Lasso, if it is 0 then it is as if we had Ridge regression. Regularization strength alpha can be defined as in Ridge or Lasso. 
# MAGIC 
# MAGIC You can enforce the values of the parameters to be positive with the parameter positive = True. Such an option is also available for Lasso. 
# MAGIC 
# MAGIC For all the variations of the linear regression you can enforce it to fit the model without an intercept. This can be done by setting the parameter fit_intercept=False
# MAGIC 
# MAGIC There is an option to scale data by the norm of each feature. If normalization is applied to fitting of the model it is automatically also applied to the predict(). We can use this method instead of standard scaling done at the beginning. 
# MAGIC 
# MAGIC 
# MAGIC ### Exercise
# MAGIC 
# MAGIC - experiment with the parameters of ElasticNet. Fit the model to x_train_lasso, y_train with different set of options, e.g.
# MAGIC     - positive=True
# MAGIC     - fit_intercept=False
# MAGIC     - l1_ratio = 0, 0.5, 1
# MAGIC     - alpha = 0.001, 0.01, 0.1, 1, 10, 100
# MAGIC     - normalize=True
# MAGIC - plot coefficients with a function plot_coef to see the effect on coefficients
# MAGIC - return RMSE on train and test 

# COMMAND ----------

# Task

lr_en = ElasticNet(l1_ratio=0.5, alpha=1, positive=True, fit_intercept=False)
lr_en.fit(x_train_lasso, y_train)
plot_coef(lr_en.coef_)

y_hat_train = lr_en.predict(x_train_lasso)
y_hat_test = lr_en.predict(x_test_lasso)


rmse_train = mean_squared_error(y_train, y_hat_train, squared=False)
rmse_test = mean_squared_error(y_test, y_hat_test, squared=False)

print(f"RMSE train: {rmse_train}")
print(f"RMSE test: {rmse_test}")

# COMMAND ----------

# MAGIC %md
# MAGIC ------------------------------------------------------------------------------------------------------------
# MAGIC Some material adapted for RBI internal purposes with full permissions from original authors. [Source](https://github.com/zatkopatrik/authentic-data-science)

# COMMAND ----------


