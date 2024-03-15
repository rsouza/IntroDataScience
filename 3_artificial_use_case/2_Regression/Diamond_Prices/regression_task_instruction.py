# Databricks notebook source
# MAGIC %md
# MAGIC # Regression Notebook
# MAGIC
# MAGIC In this task, we will work on some multivariable regression problems.
# MAGIC The dataset we will be working on is a record of Diamond sales prices.
# MAGIC
# MAGIC Remember that this is an artifical use case which is supposed to serve a **Contributor level** purposes. Thereafter, you should practice skills such as **Data Preprocessing, Data Visualisation, Preparing Data for ML** and only after you receive feedback on your initial work it is recommended to try **to fit some baseline Machine Learning Model**.
# MAGIC
# MAGIC ````
# MAGIC Goal: to find out which couple factors contribute the most to the price of a diamond. Show a strong correlation between the factors that you can come up with.
# MAGIC ````
# MAGIC     
# MAGIC One thing to keep in mind is that the result that you get at the end is highly subjective. Try to have a good correlation score, be creative, and be ready to explain the reasoning behind your work.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Dataset Information
# MAGIC We have one `.csv` file inside the ``Data`` folder prepared for you.
# MAGIC     
# MAGIC Below is an explanation of our variables from the dataset taken directly from the [dataset source](https://www.kaggle.com/datasets/shivam2503/diamonds):
# MAGIC
# MAGIC ### Diamonds's attributes:
# MAGIC - `carat` weight of the diamond (0.2-5.01)
# MAGIC - `cut` quality of the cut (Fair, Good, Very Good, Premium, Ideal)
# MAGIC - `color` diamond colour, from J (worst) to D (best)
# MAGIC - `clarity` a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
# MAGIC - `x` length in mm (0-10.74)
# MAGIC - `y` width in mm (0-58.9)
# MAGIC - `z` depth in mm (0-31.8)
# MAGIC - `depth` total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43-79)
# MAGIC - `table` width of top of diamond relative to widest point (43-95)
# MAGIC
# MAGIC ### Output variable (desired target):
# MAGIC - `price` price in US dollars ($326-$18,823)

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
# MAGIC zf = zipfile.ZipFile('Data/diamonds.zip') 
# MAGIC df = pd.read_csv(zf.open('diamonds.csv'))
# MAGIC ````
# MAGIC This is specific to our repository.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Task: Regression
# MAGIC The steps below are served as a **guide** to solve this problem. They are by no means a must or the only way to solve this partcular dataset. Feel free to use what you have learned in the previous classrooms and to be creative. Try to find out your own approach to this problem.
# MAGIC
# MAGIC
# MAGIC **Step 1: Data Loading & Describing**
# MAGIC - load the dataset into your Python Notebook
# MAGIC - convert the dataset to the desired format that you want to work with (dataframe, numpy.array, list, etc.)
# MAGIC - explore the dataset
# MAGIC - observe the variables carefully, and try to understand each variable and its meaning
# MAGIC
# MAGIC **Step 2: Data Visualisation & Exploration**
# MAGIC - employ various visualization techniques to understand the data even more thorough
# MAGIC - with the visualization tools, understand what is happening in the dataset
# MAGIC
# MAGIC **Step 3: Data Modelling**
# MAGIC - separate variables & labels
# MAGIC - split dataset into training & testing dataset
# MAGIC - pick one data modelling approach respectively the Python modelling package that you would like to use
# MAGIC - fit the training dataset to the model and train the model
# MAGIC - output the model 
# MAGIC - make prediction on testing dataset
# MAGIC
# MAGIC **Step 4: OPTIONAL Fine Tuning the Model For a Better Result**
# MAGIC - map the prediction of the testing dataset against real numbers from your dataset and compare the result
# MAGIC - make adjustments on your model for a better result (but make sure don't overfit the model)
# MAGIC     
# MAGIC **Step 5: Result Extration & Interpretation**
# MAGIC - make your conclusions and interpretation on the model and final results
# MAGIC - evaluate the performance of your model and algorithm using different KPIs <br>
# MAGIC - `bonus: ` use more visualization techniques to demonstrate the correlation between one or more variables to the happiness score
# MAGIC - `bonus: ` as well as the difference between your prediction and the actual score
# MAGIC
# MAGIC **Note!** Important criteria for evaluating your use case are well-documented cells, a good structure of the notebook with headers which are depicting various parts of it, and short comments on each part with reflections and insights that you gained.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Additional Resources: 
# MAGIC
# MAGIC **Packages that might be useful for you:**
# MAGIC - pandas: https://pandas.pydata.org/pandas-docs/stable/reference/index.html
# MAGIC - numpy: https://numpy.org/doc/
# MAGIC - scikit-learn: https://scikit-learn.org/stable/
# MAGIC - plotly: https://plotly.com/python-api-reference/
# MAGIC - lightGBM: https://lightgbm.readthedocs.io/en/latest/
# MAGIC - seaborn: https://seaborn.pydata.org/api.html
# MAGIC
# MAGIC **Useful Links:**
# MAGIC - Die Pipeline: https://wiki.rbinternational.com/confluence/display/AAT/MGF+-+Die+Pipeline
# MAGIC - Scikit homepage: https://scikit-learn.org/stable/
# MAGIC - https://scikit-learn.org/
# MAGIC - https://seaborn.pydata.org/
# MAGIC - https://plotly.com/python/
# MAGIC - https://matplotlib.org/
# MAGIC - https://medium.com/pursuitnotes/multiple-linear-regression-model-in-7-steps-with-python-c6f40c0a527
# MAGIC - https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

# COMMAND ----------

# MAGIC %md
# MAGIC Dataset citation: https://www.kaggle.com/datasets/shivam2503/diamonds/code
