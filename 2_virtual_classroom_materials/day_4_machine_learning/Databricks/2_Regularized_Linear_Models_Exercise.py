# Databricks notebook source
# MAGIC %md
# MAGIC # Regularized Linear Models

# COMMAND ----------

from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (20,15)
sns.set_theme(style="whitegrid")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data
# MAGIC
# MAGIC We load the [Boston housing data](http://lib.stat.cmu.edu/datasets/boston) and split it into train and test data. 
# MAGIC As in the last notebook, we generate [polynomial features](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
# MAGIC of the second degree.
# MAGIC We will work further with `X_train_poly`, `y_train`, `X_test_poly` and `y_test`. 
# MAGIC Run the cell below.
# MAGIC
# MAGIC ### Ethical considerations
# MAGIC
# MAGIC The dataset, which we are using in this exercise, has an ethical problem.
# MAGIC A thorough discussion of the issues can be found in [this article](https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8).  
# MAGIC The key take aways are, that there is a attribute called 'B' in the data.
# MAGIC The
# MAGIC [original authors](https://www.researchgate.net/publication/4974606_Hedonic_housing_prices_and_the_demand_for_clean_air)
# MAGIC of the dataset engineered this feature assuming that racial self-segregation has a positive impact on house prices.
# MAGIC Such an attribute furthers systemic racism and must not be used outside of educational purposes.
# MAGIC

# COMMAND ----------

# The data set is originally downloaded from  "http://lib.stat.cmu.edu/datasets/boston".

raw_df = pd.read_csv('../../../Data/Boston.csv')

y = raw_df['target']
X = pd.DataFrame(raw_df.iloc[:,1:-1])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

poly = PolynomialFeatures(2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
# depending on the version of sklearn, this will cause an error
# in that case, replace "get_feature_names_out" with "get_feature_names"
poly_names = poly.get_feature_names_out()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise
# MAGIC
# MAGIC How many features are there in total?

# COMMAND ----------

# Task 1


# COMMAND ----------

# MAGIC %md
# MAGIC We will further use the user-defined function `plot_coef` that takes as input coefficients as output of the fitted model. It plots the coefficient values and calculates average.

# COMMAND ----------

def plot_coef(lr_coef, names=[], ordered=True, hide_zero=False, figsize=(12,20)):
    """
    The function plots coefficients' values from the linear model.
    --------
    params:
        lr_coef: coefficients as they are returned from the classifier's attributes
        names: names for the coefficients, if left empty x0, x1, ... will be used
        ordered: order the coefficients according to their value
        hide_zero: hide all coefficients which are equal to 0
        figsize: tuple spcifying the size of the plot
    """
    if len(names) < 1:
        names = [f"x{i}" for i in range(len(lr_coef))]

    named_coef = pd.DataFrame({"attr": names, "coef": lr_coef})
    
    if hide_zero:
        named_coef = named_coef[named_coef["coef"] != 0]

    if ordered:
        named_coef.sort_values(by="coef", ascending=True, inplace=True)

    fig, ax = plt.subplots(figsize=figsize)

    ax.axvline(x=0, c="orange", ls="--")
    ax.scatter(x="coef", y="attr", data=named_coef)
    ax.margins(y=0.01)
    ax.set_title("Coefficients' values")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit linear regression without regularization
# MAGIC
# MAGIC ### Exercise
# MAGIC
# MAGIC - Instantiate a [linear regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) under the variable `lr`.
# MAGIC - Fit `lr` to `X_train_poly`, `y_train `.
# MAGIC - Predict with `lr` on `X_train_poly` and store the results to `y_hat_train`.
# MAGIC - Predict with `lr` on `X_test_poly` and store the results to `y_hat_test`.
# MAGIC - Return the RMSE for `y_hat_train` as well as for `y_hat_test`. 
# MAGIC
# MAGIC How do you interpret the difference in performance of the model on train and on test dataset? Can you tell if the model overfits/underfits?

# COMMAND ----------

# Task 2

lr = ...
...

y_hat_train = ...
y_hat_test = ...

print(f"RMSE train: {mean_squared_error(..., ..., squared=False)}")
print(f"RMSE test: {mean_squared_error(..., ..., squared=False)}")

# COMMAND ----------

# MAGIC %md
# MAGIC The RMSE is almost twice as big for the test set than for the train set. This suggests overfitting and a poor generalization power of the model.
# MAGIC
# MAGIC We use the function `plot_coef` on the coefficients of the fitted model to see the values of the coefficients.

# COMMAND ----------

plot_coef(lr.coef_, poly_names)

# COMMAND ----------

# MAGIC %md
# MAGIC The error values on train and test suggest that we deal here with overfitting of the model on the given set of polynomial features. 
# MAGIC We should therefore use **regularization**. 
# MAGIC
# MAGIC ## Standardization
# MAGIC
# MAGIC Before fitting any regularized model, the scaling of the features is crucial.
# MAGIC Otherwise the regularization would not be fair to features of different scales.
# MAGIC Regularized linear models assume that the inputs to the model have a zero mean and a variance in the same magnitude.
# MAGIC [`StandarScaler()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
# MAGIC deducts the mean and divides by the standard deviation. 
# MAGIC
# MAGIC ### Exercise
# MAGIC
# MAGIC - Instantiate
# MAGIC [`StandardScaler()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
# MAGIC under the name `scaler`.
# MAGIC - Apply the `fit_transform` method with the input `X_train_poly` to `scaler` and store the result into `X_train_scaled`.
# MAGIC - Once the scaler is fit to `X_train_poly` you can directly transform `X_test_poly` and store it in the variable `X_test_scaled`. You never want to fit on a test sample, because that way information from the test data might leak. Test data serves only for evaluation.

# COMMAND ----------

# Task 3

scaler = ...
X_train_scaled = ...
X_test_scaled = ...

# COMMAND ----------

# MAGIC %md
# MAGIC If you applied the standardization correctly you should see on the bottom chart the distributions of all the features concentrated around zero with similar ranges of deviation.

# COMMAND ----------

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(12, 20))

axs[0].boxplot(X_train_poly, vert=False, labels=poly_names)
axs[0].set_title('Original polynomial features')

axs[1].boxplot(X_train_scaled, vert=False, labels=poly_names)
axs[1].set_title('Scaled features')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lasso
# MAGIC Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
# MAGIC
# MAGIC ### Exercise
# MAGIC - Instantiate a Lasso regression under the name `lr_l`.
# MAGIC - Fit the model to `X_train_scaled` and `y_train`.
# MAGIC - Predict on `X_train_scaled` and `X_test_scaled` and store the predictions in `y_hat_train` and `y_hat_test`, respectively.
# MAGIC
# MAGIC Did the overfit change?

# COMMAND ----------

# Task 4


from sklearn.linear_model import Lasso

lr_l = ...
...

y_hat_train = ...
y_hat_test = ...

print(f"RMSE train: {mean_squared_error(..., ..., squared=False)}")
print(f"RMSE test: {mean_squared_error..., ..., squared=False)}")

# COMMAND ----------

# MAGIC %md
# MAGIC The performance seems to be comparable on train and test dataset. Hence, the model's generalization power is better now.
# MAGIC
# MAGIC ### Exercise
# MAGIC
# MAGIC Use `plot_coef()` on the coefficients of the lasso model.

# COMMAND ----------

# Task 5



# COMMAND ----------

# MAGIC %md
# MAGIC The average value of the coefficients is much smaller now. Also, many of the coefficients are equal to 0.

# COMMAND ----------

print(f'After applying Lasso on polynomial scaled features we remain with {np.sum(lr_l.coef_!=0)} variables.')
print('\nThe selected variables are:\n- ', end="")
print("\n- ".join(poly_names[lr_l.coef_ != 0]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise
# MAGIC
# MAGIC - Take the subset of `X_train_scaled` with only those variables that have a non-zero coefficient and store it in the variable `X_train_lasso`
# MAGIC - Do the same selection on `X_test_scaled` and save it to `X_test_lasso`.
# MAGIC - How many variables are remaining? Check it with the cell above.

# COMMAND ----------

# Task 6

X_train_lasso = ...
X_test_lasso = ...
...

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ridge
# MAGIC https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
# MAGIC
# MAGIC We have effectively performed a feature selection with Lasso. Now we will compare it to Ridge regression.
# MAGIC
# MAGIC Let's try different values for the strength of the optimization, alpha. By default it is equal to 1 and it must be a positive value. Larger values specify stronger regularization. Alpha can be set also in Lasso and Elastic Net.
# MAGIC
# MAGIC ### Exercise
# MAGIC - Fit the ridge regression to `X_train_scaled` and `y_train` with the values of alpha being 0.001, 0.01, 0.1, 1, 10 and 100 to see the effect of the regularization strength.
# MAGIC - Return the RMSE for `X_train_scaled` and `X_test_scaled` for each of the alpha options.
# MAGIC - Visulaize both RMSE curves.
# MAGIC Are you able to find the ranges where the model is over- or underfitted?

# COMMAND ----------

# Task 7

rmses = pd.DataFrame(columns=["alpha", "train", "test"])
alphas = [10**i for i in range(-3, 3)]

for alpha in alphas:    
    lr_r = ...
    ...

    y_hat_train = ...
    y_hat_test = ...

    rmse_train = mean_squared_error(..., ..., squared=False)
    rmse_test = mean_squared_error(..., ..., squared=False)
    rmses = pd.concat([rmses, pd.DataFrame([{"alpha": alpha, "train": rmse_train, "test": rmse_test}])], axis=0, ignore_index=True)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot("alpha", "train", data=rmses, label="RMSE for train set", c="b", ls="--")
ax.plot("alpha", "test", data=rmses, label="RMSE for test set", c="r")

ax.set_xscale("log")
ax.legend()
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("RMSE")

plt.show();

# COMMAND ----------

# MAGIC %md
# MAGIC In the above plot, we can observe a clear trend in the training data: as the regularization parameter \\(\alpha\\) increases, the Root Mean Square Error (RMSE) also increases monotonically. 
# MAGIC This is expected, as a higher \\(\alpha\\) imposes more restriction on the coefficients, leading to a simpler model.  
# MAGIC The more intriguing effect is seen when we look at the RMSE on the test data. 
# MAGIC As anticipated, the RMSE is high for large \\(\alpha\\) values, a phenomenon known as underfitting.
# MAGIC However, as α decreases, the RMSE starts to rise again.
# MAGIC This is because the coefficients are not sufficiently constrained, leading to an overly complex model, a situation referred to as overfitting.
# MAGIC
# MAGIC **Note:** It’s crucial not to use your test data when optimizing the hyperparameter.
# MAGIC If you aim to optimize the hyperparameter, consider using cross-validation or alternative metrics such as Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC).
# MAGIC These metrics penalize complex models and enable you to make an informed decision based solely on your training data.
# MAGIC
# MAGIC All of these observations also hold for Lasso Regression.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise
# MAGIC - Fit the model with a high value of \\(\alpha=100\\).
# MAGIC - Check how many coefficients equal 0 and plot their valuse using `plot_coef`.

# COMMAND ----------

# Task 8

lr_r_high = Ridge(...).fit(..., ...)
print(f"There are {(lr_r_high.coef_ == 0).sum()} coefficients equal to 0 for this model.")

...

# COMMAND ----------

# MAGIC %md
# MAGIC Even for a highly penalized Ridge regression model, all the coefficients are non zero.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Elastic Net
# MAGIC [Elastic Net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
# MAGIC is a combination of Lasso and Ridge which is defined by a parameter `l1_ratio`.
# MAGIC If it is equal to 1 the model is equivalent to Lasso, if it is 0 then it is as if we had a Ridge regression.
# MAGIC The regularization strength alpha can be defined just as in Ridge or Lasso. 
# MAGIC
# MAGIC You can enforce the values of the parameters to be positive with the parameter `positive = True`.
# MAGIC Such an option is also available for Lasso. 
# MAGIC
# MAGIC For all the variations of the linear regression you can enforce it to fit the model without an intercept.
# MAGIC This can be done by setting the parameter `fit_intercept=False`.
# MAGIC If `False` the data is assumed to be already centered.
# MAGIC
# MAGIC There is an option to scale data by the norm of each feature.
# MAGIC If normalization is applied to fitting of the model it is automatically also applied to the `predict()`.
# MAGIC We can use this method instead of standard scaling done at the beginning. 
# MAGIC
# MAGIC ### Exercise
# MAGIC
# MAGIC Experiment with the parameters of `ElasticNet()`.
# MAGIC Fit the model to `X_train_scaled` and `y_train` with different set of options, e.g.
# MAGIC - `positive=False`
# MAGIC - `l1_ratio = 0`, `0.5`, `1`
# MAGIC - `alpha = 0.001`, `0.01`, `0.1`, `1`, `10`, `100` 
# MAGIC
# MAGIC Plot the coefficients with `plot_coef` to see the effect on the options.
# MAGIC Return the RMSE on train and test set.

# COMMAND ----------

# Task 9



# COMMAND ----------

# MAGIC %md
# MAGIC Material adapted for RBI internal purposes with full permissions from original authors. [Source](https://github.com/zatkopatrik/authentic-data-science)
