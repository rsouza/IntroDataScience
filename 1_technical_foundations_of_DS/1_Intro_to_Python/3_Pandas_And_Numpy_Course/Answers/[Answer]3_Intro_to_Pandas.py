# Databricks notebook source
# MAGIC %md
# MAGIC # About This Notebook
# MAGIC In this **Introduction to pandas** chapter, we will learn:
# MAGIC - How pandas and NumPy combine to make working with data easier.
# MAGIC - About the two core pandas types: series and dataframes.
# MAGIC - How to select data from pandas objects using axis labels.
# MAGIC ***
# MAGIC ## 1. Understanding Pandas and NumPy
# MAGIC 
# MAGIC The pandas library provides solutions to a lot of problems. However, pandas is not so much a replacement for NumPy but serves more as an extension of NumPy. Pandas uses the the NumPy library extensively, and you will notice this more when you dig deeper into the concept.
# MAGIC 
# MAGIC The primary data structure in pandas is called a **dataframe**. 
# MAGIC This is the pandas equivalent of a Numpy 2D ndarray but with some key differences:
# MAGIC 
# MAGIC > Axis values can have string **labels**, not just numeric ones. This means that the columns can now have their own meaningful names.
# MAGIC 
# MAGIC > Dataframes can contain columns with **multiple data types**: including ``integer``, ``float``, and ``string``. This enables us to store, for example, strings and integers in one dataframe.
# MAGIC 
# MAGIC ## 2. Introduction to the Data
# MAGIC In this chapter, we will work with a data set from Fortune magazine's 2017 Global 500 list.
# MAGIC 
# MAGIC The data set is stored in a CSV file called **f500.csv**. Here is a data dictionary for some of the columns in the CSV:
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
# MAGIC 
# MAGIC After getting to know our data set, how do we actually import pandas library in Python?
# MAGIC To import pandas, we simply type in the following code:
# MAGIC 
# MAGIC ````python
# MAGIC import pandas as pd
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC Pandas' dataframes have a `.shape` attribute which returns a tuple representing the dimensions of each axis of the object. Now we want to use this and Python's `type()` function to take a closer look at the `f500` dataframe.
# MAGIC 
# MAGIC ### Task 3.3.2:
# MAGIC 
# MAGIC 1. Use Python's `type()` function to assign the type of `f500` to `f500_type`.
# MAGIC 2. Use the `DataFrame.shape` attribute to assign the shape of `f500` to `f500_shape`.
# MAGIC 3. Print both the `f500_type` and `f500_shape`.

# COMMAND ----------

import pandas as pd
f500 = pd.read_csv('f500.csv',index_col=0)
f500.index.name = None
# Start your code below:

f500_type = type(f500)
f500_shape = f500.shape

print(f500_type, f500_shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Introducing DataFrames
# MAGIC 
# MAGIC Remember how we spent so much time in the course "Be Around of Data Science" talking about rectangular data structures? Moreover, we discussed flat tables consisting of rows (observations) and columns (features)? Now this will come in really handy! 
# MAGIC 
# MAGIC I want to show you the `DataFrame.head` method. By default, it will return the first five rows of our dataframe. However, it also accepts an optional integer parameter, which specifies the number of rows:

# COMMAND ----------

f500.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC There is also the `DataFrame.tail` method to show us the last rows of our dataframe:

# COMMAND ----------

f500.tail(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.3.3:
# MAGIC 
# MAGIC 1. Use the `head()` method to select the **first 6 rows**. Assign the result to `f500_head`.
# MAGIC 2. Use the `tail()` method to select the **last 8 rows**. Assign the result to `f500_tail`.

# COMMAND ----------

# Start your code here:

f500_head = f500.head(6)
f500_tail = f500.tail(8)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Introducting DataFrames Continued
# MAGIC 
# MAGIC Now let's talk about the `DataFrame.dtypes` attribute. The `DataFrame.dtypes` attribute returns information about the types of each column.

# COMMAND ----------

print(f500.dtypes)

# COMMAND ----------

# MAGIC %md
# MAGIC To see a comprehensive overview of all the dtypes used in our dataframe, as well its shape and other information, we should use the `DataFrame.info()` [method](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.info.html). Remember that `DataFrame.info()` only prints the information, instead of returning it, so we can't assign it to a variable.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.3.4:
# MAGIC 
# MAGIC 1. Use the `DataFrame.info()` method to display information about the `f500` dataframe.

# COMMAND ----------

# Start your code below:

f500.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Selecting a Column From a DataFrame by Label (IMPORTANT)
# MAGIC 
# MAGIC Do you know that our axes in pandas all have labels and we can select data using just those labels? The **DataFrame.loc[]** attribute is exactly the syntax designed for this purpose:
# MAGIC 
# MAGIC ````python
# MAGIC df.loc[row_label, column_label]
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC Pay close attention that we use brackets ``[]`` instead of parentheses ``()`` when selecting by location.
# MAGIC 
# MAGIC Now let's look at an example:

# COMMAND ----------

f500.loc[:,"rank"]

# COMMAND ----------

# MAGIC %md
# MAGIC Notice we used `:` to specify that all rows should be selected. And also pay attention that the new dataframe has the same row labels as the original.
# MAGIC 
# MAGIC The following shortcut can also be used to select a single column:

# COMMAND ----------

rank_col = f500["rank"]
print(rank_col)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.3.5:
# MAGIC 
# MAGIC 1. Select the `industry` column. Assign the result to the variable name `industries`.
# MAGIC 2. Use Python's `type()` function to assign the type of `industries` to `industries_type`.

# COMMAND ----------

# Start your code below:

industries = f500["industry"]
industries_type = type(industries)
print(industries_type)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Selecting Columns From a DataFrame by Label Continued
# MAGIC Below, we use a list of labels to select specific columns:

# COMMAND ----------

f500.loc[:,["country", "rank"]]

# COMMAND ----------

f500[["country","rank"]]

# COMMAND ----------

# MAGIC %md
# MAGIC The code `f500.loc[:,["country", "rank"]]` and `f500[["country","rank"]]` eventually return us the same result. 

# COMMAND ----------

# MAGIC %md
# MAGIC You see that the object returned is two-dimensional, we know it's a dataframe, not a series. So instead of `df.loc[:,["col1","col2"]]`, we can also use `df[["col1", "col2"]]` to select specific columns.
# MAGIC 
# MAGIC Last but not least, let's finish by using **a slice object with labels** to select specific columns:

# COMMAND ----------

f500.loc[:,"rank":"profits"]

# COMMAND ----------

# MAGIC %md
# MAGIC The result is again a dataframe object with all of the columns from the first up until the last column in our slice. Unfortunately, there is no shortcut for selecting column slices.
# MAGIC 
# MAGIC See the table below for a short summary of techniques that we have just encountered:
# MAGIC 
# MAGIC |Select by Label|Explicit Syntax|Common Shorthand|
# MAGIC |--|--|--|
# MAGIC |Single column|df.loc[:,"col1"]|df["col1"]|
# MAGIC |List of columns|df.loc[:,["col1", "col7"]]|df[["col1", "col7"]]|
# MAGIC |Slice of columns|df.loc[:,"col1":"col4"]|

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.3.6:
# MAGIC 
# MAGIC 1. Select the `country` column. Assign the result to the variable name `countries`.
# MAGIC 2. In order, select the `revenues` and `years_on_global_500_list` columns. Assign the result to the variable name `revenues_years`.
# MAGIC 3. In order, select all columns from `ceo` up to and including `sector`. Assign the result to the variable name `ceo_to_sector`.

# COMMAND ----------

# Start your code below:

countries = f500["country"]
revenues_years = f500[["revenues","years_on_global_500_list"]]
ceo_to_sector = f500.loc[:,"ceo":"sector"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Selecting Rows From a DataFrame by Label
# MAGIC 
# MAGIC Now, let's learn how to use the labels of the index axis to select rows.
# MAGIC 
# MAGIC We can use the same syntax to select rows from a dataframe as we already have done for columns:
# MAGIC 
# MAGIC ````python
# MAGIC df.loc[row_label, column_label]
# MAGIC ````
# MAGIC 
# MAGIC #### To select a single row:

# COMMAND ----------

single_row = f500.loc["Sinopec Group"]
print(type(single_row))
print(single_row)

# COMMAND ----------

# MAGIC %md
# MAGIC A **series** is returned because it is one-dimensional. 
# MAGIC > In short, ``series`` is a one-dimensional labeled array capable of holding data of any type (integer, string, float, python objects, etc.). 
# MAGIC There are many data types that may need to be stored in this series, like integer, float, and string values, so pandas uses the **object** dtype, since none of the numeric types could cater for all values.
# MAGIC 
# MAGIC **To select a list of rows:**

# COMMAND ----------

list_rows = f500.loc[["Toyota Motor", "Walmart"]]
print(type(list_rows))
print(list_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC **To select a slice object with labels:**
# MAGIC 
# MAGIC For selection using slices, a shortcut can be used like below. This is the reason we can't use this shortcut for columns – because it's reserved for use with rows:

# COMMAND ----------

slice_rows = f500["State Grid":"Toyota Motor"]
print(type(slice_rows))
print(slice_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.3.7: 
# MAGIC 
# MAGIC By selecting data from `f500`:
# MAGIC 1. Create a new variable `toyota`, with:
# MAGIC     - Just the row with index `Toyota Motor`.
# MAGIC     - All columns.
# MAGIC 2. Create a new variable, `drink_companies`, with:
# MAGIC     - Rows with indicies `Anheuser-Busch InBev, Coca-Cola, and Heineken Holding`, in that order.
# MAGIC     - All columns.
# MAGIC 3. Create a new variable, `middle_companies` with:
# MAGIC     - All rows with indicies from `Tata Motors` to `Nationwide`, inclusive.
# MAGIC     - All columns from `rank` to `country`, inclusive.

# COMMAND ----------

# Start your code below:

toyota = f500.loc['Toyota Motor',]
drink_companies = f500.loc[["Anheuser-Busch InBev", "Coca-Cola", "Heineken Holding"]]
middle_companies = f500.loc["Tata Motors":"Nationwide", "rank":"country"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Value Counts Method
# MAGIC 
# MAGIC We understand that **series** and **dataframes** are two distinct objects. They each have their own unique methods.
# MAGIC 
# MAGIC First, let's select just one column from the ``f500`` dataframe:

# COMMAND ----------

sectors = f500["sector"]
print(type(sectors))

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we want to substitute ``Series`` in `Series.value_counts()` with the name of our `sectors` series, like this:

# COMMAND ----------

sectors_value_counts = sectors.value_counts()
print(sectors_value_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC You see that each unique non-null value is being counted and listed in the output above.
# MAGIC 
# MAGIC Well, what happens when we try to use the `Series.value_counts()` method with a dataframe? First step, we should select the `sector` and `industry` columns to create a dataframe named `sectors_industries`, like this:

# COMMAND ----------

sectors_industries = f500[["sector", "industry"]]
print(type(sectors_industries))

# COMMAND ----------

# MAGIC %md
# MAGIC Then, we'll try to use the `value_counts()` method:

# COMMAND ----------

si_value_counts = sectors_industries.value_counts()
print(si_value_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC We see that we have got the following error:
# MAGIC 
# MAGIC > AttributeError: 'DataFrame' object has no attribute 'value_counts'
# MAGIC 
# MAGIC 
# MAGIC That is because ``value_counts()`` is only a method for Series, not DataFrame.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.3.8
# MAGIC 
# MAGIC 1. Find the counts of each unique value in the `country` column in the `f500` dataframe.
# MAGIC     - Select the `country` column in the `f500` dataframe. Assign it to a variable named `countries*`
# MAGIC     - Use the `Series.value_counts()` method to return the value counts for `countries`. Assign the results to `country_counts`.

# COMMAND ----------

# Start your code below:

countries = f500["country"]
country_counts = countries.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC Below is a summary table of all the different label selection we have learned so far:
# MAGIC 
# MAGIC |Select by Label|Explicit Syntax|Shorthand Convention|
# MAGIC |-------------|-----------|---------------|
# MAGIC |Single column from dataframe|df.loc[:,"col1"]|df["col1"]|
# MAGIC |List of columns from dataframe|df.loc[:,"col1","col7"]|df["col1","col7"]|
# MAGIC |Single column from dataframe|df.loc[:,"col1":"col4"]| |
# MAGIC |Single row from dataframe| df.loc["row4"]|   |
# MAGIC |List of rows from dataframe |df.loc["row1","row8"]]|  |
# MAGIC |Slice of rows from dataframe|df.loc["row3":"row5"]| df["row3":"row5"]|
# MAGIC |Single item from series|s.loc["item8"]|s["item8"]|
# MAGIC |List of items from series|s.loc[["item1", "item7"]]|s[["item1","item7"]]|
# MAGIC |Slice of items from series|s.loc["item2":"item4"] |s["item2":"item4"]|
