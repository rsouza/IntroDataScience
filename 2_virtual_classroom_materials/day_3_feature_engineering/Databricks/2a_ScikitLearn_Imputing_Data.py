# Databricks notebook source
# MAGIC %md
# MAGIC # Simple examples of data imputation with scikit-learn
# MAGIC #### (read and play)

# COMMAND ----------

import numpy as np
import pandas as pd
from io import StringIO

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Creating some data with missing values

# COMMAND ----------

csvdata = '''
A,B,C,D,E
1,2,3,4,
5,6,,8,
0,,11,12,13
,4,15,16,17
'''

df = pd.read_csv(StringIO(csvdata))
df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Deleting missing values
# MAGIC Radical choice: [delete whole column](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html)

# COMMAND ----------

df.drop(["E"], axis=1, inplace=True)
df

# COMMAND ----------

# MAGIC %md
# MAGIC Recreating

# COMMAND ----------

df = pd.read_csv(StringIO(csvdata))
df

# COMMAND ----------

# MAGIC %md
# MAGIC Less Radical: [delete rows](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html) with missing values on "C" column

# COMMAND ----------

df.dropna(axis=0, how='any', subset=["C"], inplace=True)
df

# COMMAND ----------

# MAGIC %md
# MAGIC If you do not specify the columns, it will delete every row with any missing value

# COMMAND ----------

df.dropna(axis=0, how='any', subset=None, inplace=True)
df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Filling missing values using Pandas
# MAGIC Fill missing values with panda's [`fillna()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html) method.

# COMMAND ----------

df = pd.read_csv(StringIO(csvdata))

# COMMAND ----------

# MAGIC %md
# MAGIC Impute a constant value

# COMMAND ----------

df.fillna(value=200, inplace = True)
df

# COMMAND ----------

# MAGIC %md
# MAGIC Impute a constant value for each column

# COMMAND ----------

df = pd.read_csv(StringIO(csvdata))

# COMMAND ----------

df.fillna(value={"A": 100, "B": 200, "C": 300, "D": 400, "E": 500}, inplace = True)
df

# COMMAND ----------

# MAGIC %md
# MAGIC [Forward fill](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ffill.html) propagates the last valid observation.
# MAGIC Alternatively [`fillna(mehtod="ffill")`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html) can be used.

# COMMAND ----------

df = pd.read_csv(StringIO(csvdata))

# COMMAND ----------

df.ffill(inplace=True)
df

# COMMAND ----------

# MAGIC %md
# MAGIC [Back fill](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.bfill.html)
# MAGIC uses the next valid observation to fill missing values.
# MAGIC Again the alternative is [`.fillna(method="bfill")`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.bfill.html).

# COMMAND ----------

df = pd.read_csv(StringIO(csvdata))

# COMMAND ----------

df.bfill(inplace=True)
df

# COMMAND ----------

# MAGIC %md
# MAGIC Forward fill and back fill can be combined.

# COMMAND ----------

df = pd.read_csv(StringIO(csvdata))

# COMMAND ----------

df = df.ffill().bfill()
df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Imputing with scikit-learn
# MAGIC ### 4.1 [Simple imputing](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer)

# COMMAND ----------

from sklearn.impute import SimpleImputer

# COMMAND ----------

# MAGIC %md
# MAGIC Imputing mean values

# COMMAND ----------

df = pd.read_csv(StringIO(csvdata))

# COMMAND ----------

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(df["C"].values.reshape(-1,1))
df["C"] = imp.transform(df["C"].values.reshape(-1,1))
df

# COMMAND ----------

# MAGIC %md
# MAGIC Imputing a constant value

# COMMAND ----------

df = pd.read_csv(StringIO(csvdata))

# COMMAND ----------

imp = SimpleImputer(missing_values=np.nan, fill_value=200, strategy='constant')
imp.fit(df["C"].values.reshape(-1,1))
df["C"] = imp.transform(df["C"].values.reshape(-1,1))
df

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 [Interactive imputing](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html) (experimental)

# COMMAND ----------

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# COMMAND ----------

df = pd.read_csv(StringIO(csvdata))

# COMMAND ----------

imp_mean = IterativeImputer(random_state=0)
imp_mean.fit(df)
columns = df.columns
df = pd.DataFrame(imp_mean.transform(df), columns=columns)
df
