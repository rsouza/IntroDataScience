# Databricks notebook source
# MAGIC %md
# MAGIC # About This Notebook
# MAGIC In this **Exploring Data with Pandas: Fundamentals** chapter, we will learn:
# MAGIC - How to select data from pandas objects using boolean arrays.
# MAGIC - How to assign data using labels and boolean arrays.
# MAGIC - How to create new rows and columns in pandas.
# MAGIC
# MAGIC ****
# MAGIC ## 1. Introduction to the Data
# MAGIC
# MAGIC We learned the basics of the pandas library in the previous chapter and explored some dataframes using the techniques we have learned. Just to refresh your memory:
# MAGIC > Axis values in dataframes can have **string labels**, not just numeric ones, which makes selecting data much easier.
# MAGIC
# MAGIC > Dataframes have the ability to contain columns with **multiple data types**: such as integer, float, and string.
# MAGIC
# MAGIC In this chapter, we'll learn some other ways working with data using pandas.
# MAGIC
# MAGIC This time, we will continue working with a data set from Fortune magazine's Global 500 list 2017.
# MAGIC
# MAGIC Here is a data dictionary for some of the columns in the CSV:
# MAGIC
# MAGIC - **company**: Name of the company.
# MAGIC - **rank**: Global 500 rank for the company.
# MAGIC - **revenues**: Company's total revenue for the fiscal year, in millions of dollars (USD).
# MAGIC - **revenue_change**: Percentage change in revenue between the current and prior fiscal year.
# MAGIC - **profits**: Net income for the fiscal year, in millions of dollars (USD).
# MAGIC - **ceo**: Company's Chief Executive Officer.
# MAGIC - **industry**: Industry in which the company operates.
# MAGIC - **sector**: Sector in which the company operates.
# MAGIC - **previous_rank**: Global 500 rank for the company for the prior year.
# MAGIC - **country**: Country in which the company is headquartered.
# MAGIC - **hq_location**: City and Country, (or City and State for the USA) where the company is headquartered.
# MAGIC - **employees**: Total employees (full-time equivalent, if available) at fiscal year-end.

# COMMAND ----------

import pandas as pd
import numpy as np

f500 = pd.read_csv('../../../../Data/f500.csv',index_col=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.4.1:
# MAGIC Now I have already read the dataset into a pandas dataframe and assigned it to a variable named ``f500``.
# MAGIC
# MAGIC 1. Use the ``DataFrame.head()`` method to select the first 10 rows in ``f500``. Assign the result to ``f500_head``.
# MAGIC 2. Use the ``DataFrame.info()`` method to display information about the dataframe.

# COMMAND ----------

# Start your code here:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Vectorized Operations
# MAGIC
# MAGIC Do you remember vectorized operations which we encountered in the NumPy library? Vectorized operations enable operations applied to multiple data points at once, which does not only improve our code's performance, but also enables us to write code more quickly.
# MAGIC
# MAGIC Since pandas is an extension of NumPy, it also supports vectorized operations. Just like with NumPy, we can use any of the standard Python numeric operators with series, including:
# MAGIC
# MAGIC
# MAGIC - **Addition**: `vector_a + vector_b`
# MAGIC - **Subtraction**: `vector_a - vector_b`
# MAGIC - **Multiplication**: (unrelated to the vector multiplication in linear algebra): `vector_a * vector_b`
# MAGIC - **Division**: `vecotr_a / vector_b`
# MAGIC
# MAGIC
# MAGIC ### Task 3.4.2: 
# MAGIC 1. Subtract the values in the `rank` column from the values in the ``previous_rank`` column. Assign the result to a variable ``rank_change``.

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Series Data Exploration Methods
# MAGIC
# MAGIC Just as NumPy, pandas supports many descriptive ``stats`` methods like the following:
# MAGIC - `Series.max()`
# MAGIC - `Series.min()`
# MAGIC - `Series.mean()`
# MAGIC - `Series.median()`
# MAGIC - `Series.mode()`
# MAGIC - `Series.sum()`
# MAGIC
# MAGIC Look at how you can use the stats methods below:

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC print(my_series)
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC Output:
# MAGIC ````python
# MAGIC 0    1
# MAGIC 1    2
# MAGIC 2    3
# MAGIC 3    4
# MAGIC 4    5
# MAGIC dtype: int64
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC print(my_series.sum())
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC Output:
# MAGIC ````python
# MAGIC 15
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.4.3:
# MAGIC 1. Use the `Series.max()` method to find the maximum value for the `rank_change` series. Assign the result to the variable `rank_change_max`.
# MAGIC 2. Use the `Series.min()` method to find the minimum value for the `rank_change` series. Assign the result to the variable `rank_change_min`.

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Series Describe Method
# MAGIC
# MAGIC In this session, we will learn another method called `Series.describe` This [method](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.describe.html) shows us various information such as how many non-null values are contained in the series, the average, minimum, maximum, and other statistics.
# MAGIC
# MAGIC Let's see how we can use this method:

# COMMAND ----------

assets = f500["assets"]
print(assets.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC You can see that the values in the code segment above are displayed in <b>E-notation</b>, a type of [scientific notation](https://en.wikipedia.org/wiki/Scientific_notation).
# MAGIC
# MAGIC When we use `describe()` on a column which contains non-numeric values, we will get some different statistics, like the following example shows:

# COMMAND ----------

country = f500["country"]
print(country.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC The first line of information, `count`, is the same as for numeric columns, showing us the number of non-null values. The other three statistics are described below:
# MAGIC
# MAGIC - ``unique``: Number of unique values in the series.
# MAGIC - ``top``: Most common value in the series.
# MAGIC - ``freq``: Frequency of the most common value. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.4.4
# MAGIC 1. Return a series of descriptive statistics for the rank column in ``f500``.
# MAGIC     - Select the rank column. Assign it to a variable named ``rank``.
# MAGIC     - Use the ``Series.describe()`` method to return a series of statistics for rank. Assign the result to ``rank_desc``.
# MAGIC 2. Return a series of descriptive statistics for the `previous_rank` column in `f500`.
# MAGIC     - Select the ``previous_rank`` column. Assign it to a variable named ``prev_rank``.
# MAGIC     - Use the ``Series.describe()`` method to return a series of statistics for ``prev_rank``. Assign the result to ``prev_rank_desc``.

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Method Chaining (IMPORTANT)
# MAGIC
# MAGIC Method chaining is a common syntax for invoking multiple method calls in object-oriented programming languages. Each method returns an object, allowing the calls to be chained together in a single statement without requiring variables to store the intermediate results ([Wikipedia](https://en.wikipedia.org/wiki/Method_chaining)).
# MAGIC
# MAGIC We have actually used a couple of method chainings before in our previous examples.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Without method chaining
# MAGIC ````python
# MAGIC countries = f500["country"]
# MAGIC countries_counts = countries.value_counts()
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC #### With method chaining
# MAGIC ````python
# MAGIC countries_counts = f500["country"].value_counts() 
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC From now on, we'll try to use more and more method chaining in the code. When writing code, always assess whether method chaining will make your code harder to read. If it does, it's always preferable to break the code into more than one line.
# MAGIC
# MAGIC ### Task 3.4.5
# MAGIC 1. Use `Series.value_counts()` and `Series.loc` to return the number of companies with a value of `0` of the `previous_rank` column in the `f500` dataframe. Assign the results to `zero_previous_rank`.

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Dataframe Exploration Methods
# MAGIC
# MAGIC Since series and dataframes are two distinct objects, they have their own unique methods. However, they also have methods with the same name that behave in a similar manner. Find some examples below:
# MAGIC
# MAGIC - `Series.max()` and `DataFrame.max()`
# MAGIC - `Series.min()` and `DataFrame.min()`
# MAGIC - `Series.mean()` and `DataFrame.mean()`
# MAGIC - `Series.median()` and `DataFrame.median()`
# MAGIC - `Series.mode()` and `DataFrame.mode()`
# MAGIC - `Series.sum()` and `DataFrame.sum()`
# MAGIC
# MAGIC > In contrast to series, dataframe methods require an axis parameter in order to know which axis to calculate across. You can use integers to refer to the first and second axis. Pandas dataframe methods also accept the strings ``index`` and ``columns`` for the axis parameter:

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC # When we try to calculate along the row axis
# MAGIC DataFrame.method(axis = 0)
# MAGIC # or 
# MAGIC Dataframe.method(axis = "index")
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC # When we try to calculate along the column axis
# MAGIC DataFrame.method(axis = 1)
# MAGIC # or 
# MAGIC Dataframe.method(axis = "column")
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC For a more concrete example, if we want to find the median for the **revenues** and **profits** columns in our data set, we can do the following:

# COMMAND ----------

medians = f500[["revenues", "profits"]].median(axis=0)
# we could also use .median(axis="index")
print(medians)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.4.6
# MAGIC
# MAGIC Now, it's your time to shine!
# MAGIC 1. Use the `DataFrame.max()` method to find the maximum value for only the numeric columns from `f500` (you may need to check the documentation). Assign the result to the variable `max_f500`.

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Dataframe Describe Method
# MAGIC
# MAGIC Try to see how we can use `DataFrame.max()` method below:

# COMMAND ----------

f500.max(numeric_only=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Like series objects, there is also a `DataFrame.describe()` method that we can use to explore the dataframe more efficiently. Take a look at the `DataFrame.describe()` documentation [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html).
# MAGIC
# MAGIC There are a couple of differences between the `Series.describe()` method and `DataFrame.describe()` method. For example, the `Series.describe()` method returns a series object, the `DataFrame.describe()` method returns a dataframe object.
# MAGIC
# MAGIC ### Task 3.4.7
# MAGIC
# MAGIC Now let's have some practice with the `DataFrame.describe()` method that we just learned.
# MAGIC 1. Return a dataframe of descriptive statistics for all of the numeric columns in `f500`. Assign the result to `f500_desc`.

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Assignment with pandas
# MAGIC
# MAGIC Let's start by learning assignment, starting with the following example:

# COMMAND ----------

top5_rank_revenue = f500[["rank", "revenues"]].head()
print(top5_rank_revenue)

# COMMAND ----------

top5_rank_revenue["revenues"] = 0
print(top5_rank_revenue)

# COMMAND ----------

# MAGIC %md
# MAGIC As in Numpy, we can apply the same technique that we use to select data to assignment. 
# MAGIC > Just remember, when we selected a whole column by label and used assignment, we assigned the value to every item in that column.
# MAGIC
# MAGIC When we provide labels for both axes, we assign the value to a single item within our dataframe.

# COMMAND ----------

top5_rank_revenue.loc["Sinopec Group", "revenues"] = 999
print(top5_rank_revenue)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.4.8
# MAGIC
# MAGIC Use again our Fortune 500 data set:
# MAGIC 1. The company "Dow Chemical" has named a new CEO. Update the value where the row label is `Dow Chemical` by changing the ceo column to `Jim Fitterling` in the `f500` dataframe.

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Using Boolean Indexing with pandas Objects
# MAGIC
# MAGIC In order to replace many values at the same time, we recommend using <b> boolean indexing </b> to change all rows that meet the same criteria, just like we did with NumPy.
# MAGIC
# MAGIC Let's take a look at the dataframe example below. This is a dataframe of people and their favorite numbers:

# COMMAND ----------

import pandas as pd

d = {'name': ["Kyllie", "Rahul", "Michael", "Sarah"], 'num': [12, 8, 5, 8]}

df = pd.DataFrame(data=d)
df

# COMMAND ----------

# MAGIC %md
# MAGIC If we want to check which people have a favorite number of 8, we can first perform a vectorized boolean operation that produces a boolean series:

# COMMAND ----------

num_bool = df["num"] == 8
num_bool

# COMMAND ----------

# MAGIC %md
# MAGIC We have used a series to index the whole dataframe, leaving us with the rows that correspond only to people whose favorite number is 8:

# COMMAND ----------

result = df[num_bool]
result

# COMMAND ----------

# MAGIC %md
# MAGIC You see that we didn't use ``loc[]``. The reason for this is that boolean arrays use the same shortcut as slices to select along the index axis. We can also use the boolean series to index just one column of the dataframe:

# COMMAND ----------

result = df.loc[num_bool,"name"]
result

# COMMAND ----------

# MAGIC %md
# MAGIC You see that we have used `df.loc[]` to specify both axes.
# MAGIC
# MAGIC ### Task 3.4.9
# MAGIC 1. Create a boolean series, `motor_bool`, that compares whether the values in the `industry` column from the `f500` dataframe are equal to `"Motor Vehicles and Parts"`.
# MAGIC 2. Use the `motor_bool` boolean series to index the `country` column. Assign the result to `motor_countries`.

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Using Boolean Arrays to Assign Values (OPTIONAL)
# MAGIC
# MAGIC In this session, we will look at how we can combine assignment and boolean indexing in pandas.
# MAGIC
# MAGIC In the following example, we change the `'Motor Vehicles & Parts'` values in the sector column to `'Motor Vehicles and Parts'` – i.e. we will change the ampersand (`&`) to `and`.

# COMMAND ----------

# First, we create a boolean series by comparing the values in the sector column to 'Motor Vehicles & Parts'
ampersand_bool = f500["sector"] == "Motor Vehicles & Parts"

# Next, we use that boolean series and the string "sector" to perform the assignment.
f500.loc[ampersand_bool,"sector"] = "Motor Vehicles and Parts"

# COMMAND ----------

# MAGIC %md
# MAGIC We can do what we just did as in one line: remove the intermediate step of creating a boolean series, and combine everything like this:

# COMMAND ----------

f500.loc[f500["sector"] == "Motor Vehicles & Parts","sector"] = "Motor Vehicles and Parts"

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have learned how to put everything into one line, we can use this technique and replace the values in the `previous_rank` column. We want to replace the items with value `0`, as `0` is no reasonable rank. What we can replace these values with? We can replace these values with `np.nan` – this value is used in pandas to represent values that cannot be represented numerically, such as some missing values.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.4.10
# MAGIC 1. Use boolean indexing to update values in the `previous_rank` column of the `f500` dataframe:
# MAGIC     - There should now be a value of `np.nan` where there previously was a value of `0`.
# MAGIC     - It is up to you whether you assign the boolean series to its own variable first, or whether you complete the operation in one line.
# MAGIC 2. Create a new pandas series, `prev_rank_after`, using the same syntax that was used to create the `prev_rank_before` series.
# MAGIC 3. Compare the `prev_rank_before` and the `prev_rank_after` series.

# COMMAND ----------

import numpy as np
f500 = pd.read_csv('../../../../Data/f500.csv',index_col=0)
prev_rank_before = f500["previous_rank"].value_counts(dropna=False).head()

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Creating New Columns (IMPORTANT)
# MAGIC
# MAGIC When we try to assign a value or values to a new column label in pandas, a new column will be created in our dataframe. Below we've added a new column to a dataframe named `top5_rank_revenue`:

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC top5_rank_revenue["year_founded"] = 0
# MAGIC print(top5_rank_revenue)
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC |_                          |rank |revenues |year_founded|
# MAGIC |--------------------------|-----|---------|------------|
# MAGIC |Walmart                   |   1 |       0 |           0|
# MAGIC |State Grid                |   2 |       0 |           0|
# MAGIC |Sinopec Group             |   3 |     999 |           0|
# MAGIC |China National Petroleum  |   4 |       0 |           0|
# MAGIC |Toyota Motor              |   5 |       0 |           0|

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.4.11
# MAGIC 1. Add a new column named `rank_change` to the `f500` dataframe. This column should show the change of ranks which you get by subtracting the values in the `rank` column from the values in the `previous_rank` column.
# MAGIC 2. Use the `Series.describe()` method to return a series of descriptive statistics for the `rank_change` column. Assign the result to `rank_change_desc`.
# MAGIC 3. Verify that the minimum value of the `rank_change` column is now greater than `-500`.

# COMMAND ----------

# Start your code below:

