# Databricks notebook source
# MAGIC %md
# MAGIC # Clustering Task Instruction
# MAGIC
# MAGIC In this task, we will work on some clustering problems. The dataset we will be working on is a 52-week of sales transaction report. The **ultimate goal** of this task is to find similar time series in the sales transaction data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Dataset Information
# MAGIC The variable description is taken directly from the [dataset source](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata).
# MAGIC The dataset summarizes the usage behavior of about 9000 active credit card holders during a timeframe of 6 months.
# MAGIC The file is at a customer level with 18 behavioral variables:
# MAGIC
# MAGIC - `CUST_ID`: Identification of Credit Card holder (Categorical)
# MAGIC - `BALANCE`: Balance amount left in their account to make purchases (
# MAGIC - `BALANCE_FREQUENCY`: How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
# MAGIC - `PURCHASES`: Amount of purchases made from account
# MAGIC - `ONEOFF_PURCHASES`: Maximum purchase amount done in one-go
# MAGIC - `INSTALLMENTS_PURCHASES`: Amount of purchase done in installment
# MAGIC - `CASH_ADVANCE`: Cash in advance given by the user
# MAGIC - `PURCHASES_FREQUENCY`: How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
# MAGIC - `ONEOFFPURCHASESFREQUENCY`: How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
# MAGIC - `PURCHASESINSTALLMENTSFREQUENCY`: How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
# MAGIC - `CASHADVANCEFREQUENCY`: How frequently the cash in advance being paid
# MAGIC - `CASHADVANCETRX`: Number of Transactions made with "Cash in Advanced"
# MAGIC - `PURCHASES_TRX`: Numbe of purchase transactions made
# MAGIC - `CREDIT_LIMIT`: Limit of Credit Card for user
# MAGIC - `PAYMENTS`: Amount of Payment done by user
# MAGIC - `MINIMUM_PAYMENTS`: Minimum amount of payments made by user
# MAGIC - `PRCFULLPAYMENT`: Percent of full payment paid by user
# MAGIC - `TENURE`: Tenure of credit card service for user

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
# MAGIC zf = zipfile.ZipFile('Data/CC_GENERAL.zip') 
# MAGIC df = pd.read_csv(zf.open('CC_GENERAL.csv'))
# MAGIC ````
# MAGIC This is specific to our repository.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Task Instruction
# MAGIC The steps below are served as a guidance to solve this clustering problem. They are by no means a must or the only way to solve this partcular dataset. Feel free to use what you have learned in the previous program and to be creative. Try to find out your own approach to this problem.
# MAGIC
# MAGIC **Step 1: Data loading & preprocessing**
# MAGIC - load the data into Python Notebook and convert it to the appropriate format (dataframe, numpy.array, list, etc.)
# MAGIC - observe & explore the dataset
# MAGIC - check for null values
# MAGIC
# MAGIC **Step 2: Data modelling**
# MAGIC - standardize the data to normal distribution 
# MAGIC - pick one data modelling approach respectively the Python modelling package that you would like to use
# MAGIC - fit the training dataset to the model and train the model
# MAGIC - output the model 
# MAGIC
# MAGIC **Step 3: Result extration & interpretation**
# MAGIC - make your conclusions and interpretation on the model and final results
# MAGIC - evaluate the performance of your model and algorithm using different KPIs 
# MAGIC
# MAGIC **Note!** Important criteria for evaluating your use case are well-documented cells, a good structure of the notebook with headers which are depicting various parts of it, and short comments on each part with reflections and insights that you gained.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Additional Resources:
# MAGIC    
# MAGIC **Packages that might be useful for you:**
# MAGIC - pandas
# MAGIC - numpy
# MAGIC - sklearn
# MAGIC - sklearn.cluster
# MAGIC - matplotlib 
# MAGIC
# MAGIC **Useful links:**
# MAGIC - k-means clustering: https://en.wikipedia.org/wiki/K-means_clustering
# MAGIC - hierarchical clustering: https://en.wikipedia.org/wiki/Hierarchical_clustering & https://towardsdatascience.com/understanding-the-concept-of-hierarchical-clustering-technique-c6e8243758ec
# MAGIC - the elbow method: https://en.wikipedia.org/wiki/Elbow_method_(clustering)#:~:text=In%20cluster%20analysis%2C%20the%20elbow,number%20of%20clusters%20to%20use.
# MAGIC
# MAGIC    

# COMMAND ----------

# MAGIC %md
# MAGIC **Dataset citation:**
# MAGIC https://www.kaggle.com/datasets/arjunbhasin2013/ccdata
