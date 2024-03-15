# Databricks notebook source
# MAGIC %md
# MAGIC # Classification Task Instruction
# MAGIC In this task, you will get a sense of using Python to solve a classification problem.
# MAGIC The [dataset](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction) that we will be working with is from the Taiwanese Strock Exchange about the bankruptcy of various companies.   
# MAGIC ````
# MAGIC  Goal: to predict, if a company is bankrupt (denoted in **variable y**) or not.
# MAGIC ````
# MAGIC
# MAGIC Remember that this is an artifical use case which is supposed to serve a **Contributor level** purposes. Thereafter, you should practice skills such as **Data Preprocessing, Data Visualisation, Preparing Data for ML** and only after you receive feedback on your initial work it is recommended to try **to fit some baseline Machine Learning Model**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Dataset Information
# MAGIC
# MAGIC Inside **Data** folder there is a `.zip` file containing your dataset.
# MAGIC
# MAGIC There are 95 input variables:
# MAGIC - Headers have meaningfull names
# MAGIC - Most are numeric variable
# MAGIC - A few are binary variables encoded with 0 and 1
# MAGIC
# MAGIC Output variable (desired target):
# MAGIC - `Bankrupt?` indicates wether the company is bankrupt or not(binary: 0, 1)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Importing the Dataset
# MAGIC A dataset can be imported directly from a `.zip` file.
# MAGIC To import a dataset, you will need to specify the file where is dataset is located.
# MAGIC The relative path below is correct for the location of this instruction file.
# MAGIC ````python
# MAGIC import pandas as pd
# MAGIC import zipfile
# MAGIC
# MAGIC zf = zipfile.ZipFile('Data/Company_Bankruptcy.zip') 
# MAGIC df = pd.read_csv(zf.open('Company_Bankruptcy.csv'))
# MAGIC ````
# MAGIC This is specific to our repository.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Task Instruction
# MAGIC The steps below are served as a **guide** to solve this problem. They are by no means a must or the only way to solve this partcular dataset. Feel free to use what you have learned in the previous classrooms and to be creative. Try to find out your own approach to this problem.
# MAGIC
# MAGIC **Step 1: Data Loading & Preprocessing**
# MAGIC - load the dataset into your Python Notebook
# MAGIC - convert the dataset to the desired format that you want to work with (dataframe, numpy.array, list, etc.)
# MAGIC - explore the dataset
# MAGIC - observe the variables carefully, and try to understand each variable and its business value
# MAGIC - don't forget the special treatment to null values
# MAGIC
# MAGIC
# MAGIC **Step 2: Data Visualisation & Exploration**
# MAGIC - employ various visualisation skills that you acquired inside and outside the classroom
# MAGIC - with the visualisation tools, understand what is happening in the dataset
# MAGIC
# MAGIC
# MAGIC **Step 3: Data modelling**
# MAGIC - separate variables & labels
# MAGIC - split dataset into training & testing dataset
# MAGIC - pick one data modelling approach respectively the Python modelling package that you would like to use
# MAGIC - fit the training dataset to the model and train the model
# MAGIC - output the model 
# MAGIC - make prediction on testing dataset
# MAGIC
# MAGIC
# MAGIC **OPTIONAL: Step 4: Fine tune the model or use more advanced modelling approaches**
# MAGIC - map the prediction of the testing dataset against real numbers from your dataset and compare the result
# MAGIC - make adjustments on your model for a better result (but make sure don't overfit the model)
# MAGIC
# MAGIC
# MAGIC **Step 5: Result extration & interpretation**
# MAGIC - make your conclusions and interpretation on the model and final results
# MAGIC - evaluate the performance of your model and algorithm using different KPIs
# MAGIC
# MAGIC **Note!** Important criteria for evaluating your use case are well-documented cells, a good structure of the notebook with headers which are depicting various parts of it, and short comments on each part with reflections and insights that you gained.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Additional Resources
# MAGIC - you can take a look at one of the use cases that AA tribe had done in the past and get a sense of how the modelling approach feels like: general description of the modelling approach: <https://wiki.rbinternational.corp/confluence/display/AAT/Modelling+approach+RBBG>    
# MAGIC     - random forest approach: <https://wiki.rbinternational.corp/confluence/display/AAT/Random+Forest+Modelling+Approach>
# MAGIC     - gradient boosting approach:  <https://wiki.rbinternational.corp/confluence/display/AAT/Gradient+Boosting+Modelling+Approach>
# MAGIC - some of the modelling methods that you can go over:
# MAGIC     - random forest : https://www.youtube.com/watch?v=J4Wdy0Wc_xQ
# MAGIC         - for random forest, please first understand [decision trees](https://en.wikipedia.org/wiki/Decision_tree#:~:text=A%20decision%20tree%20is%20a,only%20contains%20conditional%20control%20statements.) completely
# MAGIC     - gradient boosting: https://www.youtube.com/watch?v=3CC4N4z3GJc
# MAGIC     - support vector machine: https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
# MAGIC
# MAGIC **Packages that might be useful for you:**
# MAGIC - pandas: https://pandas.pydata.org/pandas-docs/stable/reference/index.html
# MAGIC - numpy: https://numpy.org/doc/
# MAGIC - scikit-learn: https://scikit-learn.org/stable/
# MAGIC - sklearn.linear_model: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# MAGIC - sklearn.datasets: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets
# MAGIC - sklearn.ensemble: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# MAGIC - sklearn.preprocessing: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# MAGIC - sklearn.dummy: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
# MAGIC - sklearn.metrics: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# MAGIC
# MAGIC
# MAGIC **Useful links:**
# MAGIC - Die Pipeline: https://wiki.rbinternational.com/confluence/display/AAT/MGF+-+Die+Pipeline
# MAGIC - Importing data with read_csv: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
# MAGIC - subsetting dataset in pandas: https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html
# MAGIC - column transformers with mixed types: https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
# MAGIC - feature scaling for machine learning: https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/
# MAGIC - what is ont-hot encoding: https://www.geeksforgeeks.org/ml-one-hot-encoding-of-datasets-in-python/
# MAGIC - xgboost model with python: https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
# MAGIC - random forest classifier: https://www.datacamp.com/community/tutorials/random-forests-classifier-python

# COMMAND ----------

# MAGIC %md
# MAGIC **Dataset citation:**
# MAGIC [[Liang et al., 2016] Liang, D., Lu, C.-C., Tsai, C.-F., and Shih, G.-A. (2016) Financial Ratios and Corporate Governance Indicators in Bankruptcy Prediction: A Comprehensive Study. European Journal of Operational Research, vol. 252, no. 2, pp. 561-572.](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
