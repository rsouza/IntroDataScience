# Databricks notebook source
# MAGIC %md
# MAGIC # Regression Notebook
# MAGIC
# MAGIC In this task, we will work on some multivariable regression problems. The dataset we will be working on is a world happiness report from years `2015` to `2022`. It is a survey of the state of global happiness based on local economic production, social welfares, and etc.
# MAGIC
# MAGIC Remember that this is an artifical use case which is supposed to serve a **Contributor level** purposes. Thereafter, you should practice skills such as **Data Preprocessing, Data Visualisation, Preparing Data for ML** and only after you receive feedback on your initial work it is recommended to try **to fit some baseline Machine Learning Model**.
# MAGIC
# MAGIC Some insipirations that we can get out of this dataset: 
# MAGIC - imagine that you want to create a complete new country with the goal of having the happiest citizens, which factors or some variables that you might want to pay attention the most?
# MAGIC - imagine that you are the president of your country, how can you improve the happiness level of your country? What are some of the factors that you might want to look out for?
# MAGIC
# MAGIC ````
# MAGIC Goal: to find out which couple factors contribute the most to the happiness score that we have. Show a strong correlation between the factors that you can come up with.
# MAGIC ````
# MAGIC     
# MAGIC One thing to keep in mind is that the result that you get at the end is highly subjective. Try to have a good correlation score, be creative, and be ready to explain the reasoning behind your work.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Dataset Information
# MAGIC We have 8 different files inside the ``Datasets`` folder prepared for you: `2015.csv`, `2016.csv`, ..., `2022.csv`. The name of the file corresponds to the year of which this report is describing.
# MAGIC     
# MAGIC Below is an explanation of our variables from the dataset:
# MAGIC
# MAGIC ### Country's profile:
# MAGIC - `country` the respective country
# MAGIC - `region` the region that the country belongs to
# MAGIC - `happiness rank` the ranking of this country based on its happiness score
# MAGIC - `happiness score` the happiness score
# MAGIC
# MAGIC ### Independent variables contributing to happiness score:
# MAGIC  - `economy gdp per capita` the monetary value of all finished goods and services made within a country during a specific period. 
# MAGIC  - `family` the national average of the binary responses (either 0 or 1) to the Gallup World Poll (GWP) question “If you were in trouble, do you have relatives or friends you can count on to help you whenever you need them, or not?”
# MAGIC  - `health life expectancy` the expected number of remaining years of life spent in good health from a particular age, typically birth
# MAGIC  - `freedom` freedom to make life choices, the national average of binary responses to the GWP question “Are you satisfied or dissatisfied with your freedom to choose what you do with your life?”
# MAGIC  - `trust government corruption` people's perception of corruption
# MAGIC  - `generosity`  the residual of regressing the national average of GWP responses to the question “Have you donated money to a charity in the past month?” on GDP per capita <br>
# MAGIC  Explanations are taken from: [world happiness report](https://worldhappiness.report/ed/2019/changing-world-happiness/)
# MAGIC
# MAGIC ### Other attributes:
# MAGIC - `dystopia` dystopia is an imaginary country that has the world’s least-happy people. The dystopia happiness score (1.85) i.e. the score of a hypothetical country that has a lower rank than the lowest ranking country on the report, plus the residual value of each country (a number that is left over from the normalization of the variables which cannot be explained)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Importing The Dataset
# MAGIC To import a dataset, you will need to specify the **relative** path of the csv-file, for example:
# MAGIC ````python
# MAGIC 'Datasets/2015.csv'
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
# MAGIC - employ various visualization techniques to understand the data even more thoroughly
# MAGIC - `bonus: ` ex. correlation between happiness score & GDP
# MAGIC     
# MAGIC **Step 3: Data Preprocessing**
# MAGIC - harmonize the column names in all 5 files
# MAGIC - merge all 8 files into one big dataframe (or numpy.array, list, no restrictions here)
# MAGIC - speical treatment to null values 
# MAGIC
# MAGIC **Step 4: Data Modelling**
# MAGIC - separate variables & labels
# MAGIC - split dataset into training & testing dataset
# MAGIC - pick one data modelling approach respectively the Python modelling package that you would like to use
# MAGIC - fit the training dataset to the model and train the model
# MAGIC - output the model 
# MAGIC - make prediction on testing dataset
# MAGIC
# MAGIC **Step 5: OPTIONAL Fine Tuning the Model For a Better Result**
# MAGIC - map the prediction of the testing dataset against real numbers from your dataset and compare the result
# MAGIC - make adjustments on your model for a better result (but make sure don't overfit the model)
# MAGIC     
# MAGIC **Step 6: Result Extration & Interpretation**
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
# MAGIC - https://medium.com/analytics-vidhya/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-6b0fe70b32d7

# COMMAND ----------

# MAGIC %md
# MAGIC Dataset citation: <https://worldhappiness.report/ed/2020/social-environments-for-world-happiness/>
