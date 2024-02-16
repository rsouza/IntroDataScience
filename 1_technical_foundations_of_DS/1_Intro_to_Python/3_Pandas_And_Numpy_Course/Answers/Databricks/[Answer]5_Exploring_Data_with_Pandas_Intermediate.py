# Databricks notebook source
# MAGIC %md
# MAGIC # About This Notebook
# MAGIC In this last chapter of the course, **Exploring Data with pandas: Intermediate**, we will learn:
# MAGIC - Select columns, rows and individual items using their integer location.
# MAGIC - Use `pd.read_csv()` to read CSV files in pandas.
# MAGIC - Work with integer axis labels.
# MAGIC - How to use pandas methods to produce boolean arrays.
# MAGIC - Use boolean operators to combine boolean comparisons to perform more complex analysis.
# MAGIC - Use index labels to align data.
# MAGIC - Use aggregation to perform advanced analysis using loops.
# MAGIC ***
# MAGIC ## 1. Reading CSV files with pandas (IMPORTANT)
# MAGIC
# MAGIC In the previous notebook about the fundamentals of exploring data with pandas, we worked with Fortune Global 500 dataset. In this chapter, we will learn how to use the `pandas.read_csv()` function to read in CSV files.

# COMMAND ----------

# MAGIC %md
# MAGIC Previously, we used the snippet below to read our CSV file into pandas.

# COMMAND ----------

import pandas as pd
f500 = pd.read_csv("../../../../../Data/f500.csv", index_col=0)
f500.index.name = None
f500.head()

# COMMAND ----------

# MAGIC %md
# MAGIC But if you look closely, you may see that the index axis labels are the values from the first column in the data set, **company**:

# COMMAND ----------

# MAGIC %md
# MAGIC company,rank,revenues,revenue_change
# MAGIC Walmart,1,485873,0.8
# MAGIC State Grid,2,315199,-4.4
# MAGIC Sinopec Group,3,267518,-9.1
# MAGIC China National Petroleum,4,262573,-12.3
# MAGIC Toyota Motor,5,254694,7.7

# COMMAND ----------

# MAGIC %md
# MAGIC You will see that in the [`read_csv()` function](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html), the `index_col` parameter is optional from the official documentation. When we specify a value of `0`, the first column will be used as the row labels.
# MAGIC
# MAGIC Compare with the dataframe above, notice how the `f500` dataframe looks like if we remove the second line using `f500.index.name = None`.

# COMMAND ----------

f500 = pd.read_csv("../../../../../Data/f500.csv", index_col=0)
f500.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Do you see the text **company** above the index labels? This is the name of the first column in the CSV. This value is used as the **axis name** for the index axis in Pandas.
# MAGIC
# MAGIC You see that both the column and index axes can have names assigned to them. Originally, we accessed the name of the index axes and set it to `None`, that's why the dataframe didn't have a name for the index axis.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.5.1
# MAGIC 1. Use the `pandas.read_csv()` function to read the `f500.csv` CSV file as a pandas dataframe. Assign it to the variable name `f500`.
# MAGIC     - Do not use the `index_col` parameter.
# MAGIC 2. Use the following code to insert the NaN values (missing values) into the `previous_rank` column: <br>
# MAGIC ````python
# MAGIC f500.loc[f500["previous_rank"] == 0, "previous_rank"] = np.nan
# MAGIC ````
# MAGIC Remark: If you get a notice that `np` is not defined, you have to import NumPy by typing `import numpy as np`.

# COMMAND ----------

# Start your code below:

import pandas as pd
import numpy as np

f500 = pd.read_csv("../../../../../Data/f500.csv")
f500.index.name = None

f500.loc[f500["previous_rank"] == 0, "previous_rank"] = np.nan
f500_selection = f500.loc[:, ["rank", "revenues", "revenue_change"]].head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Using iloc to select by integer position
# MAGIC
# MAGIC In the previous exercise we read our CSV file into pandas. But this time, we didn't use the `index_col` parameter:

# COMMAND ----------

f500 = pd.read_csv("../../../../../Data/f500.csv")
print(f500[['company', 'rank', 'revenues']].head())

# COMMAND ----------

# MAGIC %md
# MAGIC There are two significant differences with the approach that we just took above:
# MAGIC - the **company** column is now included as a regular column, not as an index column
# MAGIC - the **index labels** now start from `0` as **integers**
# MAGIC
# MAGIC This is the more conventional way how we should read in a dataframe, and we will be going with this method from now on.
# MAGIC
# MAGIC However, do you still remember how we worked with a dataframe with **string index labels**? We used `loc[]` to select the data.

# COMMAND ----------

# MAGIC %md
# MAGIC For selecting rows and columns by their integer positions, we use `iloc[]`. Using `iloc[]` is almost identical to indexing with NumPy, with integer positions starting at `0` like ndarrays and Python lists.
# MAGIC
# MAGIC `DataFrame.iloc[]` behaves similarly to `DataFrame.loc[]`. The full syntax for `DataFrame.iloc[]`, in pseudocode, is: 
# MAGIC
# MAGIC ````python
# MAGIC df.iloc[row_index, column_index]
# MAGIC ````
# MAGIC
# MAGIC To help you memorize the two syntaxes easier:
# MAGIC - ``loc``: label based selection
# MAGIC - ``iloc``: integer position based selection

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.5.2
# MAGIC 1. Select just the fifth row of the `f500` dataframe. Assign the result to `fifth_row`.
# MAGIC 2. Select the value in the first row of the `company` column. Assign the result to `company_value`.

# COMMAND ----------

# Start your code below:

fifth_row = f500.iloc[4,:]
company_value = f500.iloc[0,0] # company is the first row

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Using iloc to select by integer position continued
# MAGIC
# MAGIC If we want to select the first column from our `f500` dataset, we need to use ``:``, a colon, to specify all rows, and then use the integer ``0`` to specify the first column, like this:

# COMMAND ----------

first_column = f500.iloc[:,0]
print(first_column)

# COMMAND ----------

# MAGIC %md
# MAGIC To specify a positional slice, try to use the same shortcut that we used with labels. Below is an example how we would select the rows between index positions one to four (inclusive):

# COMMAND ----------

second_to_fifth_rows = f500[1:5]
print(second_to_fifth_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC Pay attention to the fact that the row at index position `5` is not included, just as if we were slicing with a Python list or NumPy ndarray. Recall that `loc[]` handles slicing differently:
# MAGIC
# MAGIC - With `loc[]`, the ending slice **is** included.
# MAGIC - With `iloc[]`, the ending slice **is not** included.
# MAGIC
# MAGIC The table below summarizes the usage of `DataFrame.iloc[]` and `Series.iloc[]` to select by integer position:
# MAGIC
# MAGIC |Select by integer position| Explicit Syntax| Shorthand Convention|
# MAGIC |--|--|--|
# MAGIC |Single column from dataframe|df.iloc[:,3]| |
# MAGIC |List of columns from dataframe|df.iloc[:,[3,5,6]] | |
# MAGIC |Slice of columns from dataframe|df.iloc[:,3:7]| |
# MAGIC |Single row from dataframe|df.iloc[20]| |
# MAGIC |List of rows from dataframe|df.iloc[[0,3,8]]| |
# MAGIC |Slice of rows from dataframe|df.iloc[3:5]|df[3:5]|
# MAGIC |Single items from series|s.iloc[8]|s[8]|
# MAGIC |List of item from series |s.iloc[[2,8,1]]|s[[2,8,1]]|
# MAGIC |Slice of items from series|s.iloc[5:10]|s[5:10]|

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.5.3
# MAGIC 1. Select the first three rows of the `f500` dataframe. Assign the result to `first_three_rows`.
# MAGIC 2. Select the first and seventh rows and the first five columns of the `f500` dataframe. Assign the result to `first_seventh_row_slice`.

# COMMAND ----------

# Start your code below:

first_three_rows = f500[:3]
first_seventh_row_slice = f500.iloc[[0, 6], :5]

print(first_three_rows)
print(first_seventh_row_slice)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Using pandas methods to create boolean masks
# MAGIC
# MAGIC There are two methods that I want to introduce to you in this chapter, which are the `Series.isnull()` [method](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.isnull.html) and `Series.notnull()` [method](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.notnull.html). These two methods can be either used to select rows that contain null (or NaN) values or to select rows that do **not** contain null values.
# MAGIC
# MAGIC Let's first have a look at the `Series.isnull()` method, which is used to view rows with null values (i.e. missing values) in one column.
# MAGIC Here is an example for the `revenue_change` column:

# COMMAND ----------

rev_is_null = f500["revenue_change"].isnull()
print(rev_is_null.head())

# COMMAND ----------

# MAGIC %md
# MAGIC We see that using `Series.isnull()` resulted in a boolean series. Just like in NumPy, we can use this series to filter our dataframe, `f500`:

# COMMAND ----------

import pandas as pd
import numpy as np

f500 = pd.read_csv("../../../../../Data/f500.csv")
f500.index.name = None


rev_change_null = f500[rev_is_null]
print(rev_change_null[["company", "country","sector"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.5.4
# MAGIC 1. Use the `Series.isnull()` method to select all rows from `f500` that have a null value for the `profit_change` column. Select only the `company`, `profits`, and `profit_change` columns. Assign the result to `null_profit_change`.

# COMMAND ----------

# Start your code below:

profit_change_bool  = f500["profit_change"].isnull()
null_profit_change = f500[profit_change_bool][["company", "profits", "profit_change"]]
print(null_profit_change)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Working with Integer Labels (OPTIONAL)
# MAGIC
# MAGIC Now let's check the difference between `DataFrame.loc[]` and `DataFrame.iloc[]` – what kind of different output will they provide?:

# COMMAND ----------

# MAGIC %md
# MAGIC We can use `DataFrame.iloc[]`, and it will get us the following result:

# COMMAND ----------

# Only works if you have completed task 3.5.4

first_null_profit_change = null_profit_change.iloc[0]
print(first_null_profit_change)

# COMMAND ----------

# MAGIC %md
# MAGIC But `DataFrame.loc[]` will throw an error:

# COMMAND ----------

first_null_profit_change = null_profit_change.loc[0]

# COMMAND ----------

# MAGIC %md
# MAGIC We get an error, telling us that **the label [0] is not in the [index]**. Remember that `DataFrame.loc[]` is used for label based selection:
# MAGIC
# MAGIC - ``loc``: label based selection
# MAGIC - ``iloc``: integer position based selection
# MAGIC
# MAGIC We see that there is no row with a 0 label in the index, we got the error above. If we wanted to select a row using `loc[]`, we'd have to use the integer label for the first row — `5`.

# COMMAND ----------

first_null_profit_change = null_profit_change.loc[5]
print(first_null_profit_change)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Pandas Index Alignment (OPTIONAL)
# MAGIC Do you know that pandas has a very powerful aspect? --- Almost every operation will <b>align on the index labels</b>. Let's look at an example below to understand what this means. We have a dataframe named `food` and a series named `alt_name`:

# COMMAND ----------

import pandas as pd
d = {'fruit_veg': ["fruit", "veg", "fruit", "veg","veg"], 'qty': [4, 2, 4, 1, 2]}
food = pd.DataFrame(data=d)
food.index = ['tomato', 'carrot', 'lime', 'corn','eggplant'] 
food

# COMMAND ----------

alt_name = pd.Series(['rocket', 'aubergine', 'maize'], index=["arugula", "eggplant", "corn"])
alt_name

# COMMAND ----------

# MAGIC %md
# MAGIC By observing the two dataframes above, we see that though the `food` dataframe and the `alt_name` series have different numbers of items, they share two of the same index labels which are `corn` and `eggplant`. However, these are in different orders. If we wanted to add `alt_name` as a new column in our `food` dataframe, we can use the following code:

# COMMAND ----------

food["alt_name"] = alt_name

food

# COMMAND ----------

# MAGIC %md
# MAGIC When we perform the code above, pandas will intentionally ignore the order of the ``alt_name`` series, and automatically align on the index labels.
# MAGIC
# MAGIC In addition, Pandas will also:
# MAGIC
# MAGIC - Discard any items that have an index that doesn't match the dataframe (like `arugula`).
# MAGIC - Fill any remaining rows with `NaN`.
# MAGIC
# MAGIC Observe the result again carefully.

# COMMAND ----------

# Below is the result
food

# COMMAND ----------

# MAGIC %md
# MAGIC You see that with every occasion, the pandas library will align on index, no matter if our index labels are strings or integers - this makes working with data from different sources much much easier.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Using Boolean Operators (IMPORTANT)
# MAGIC We can combine boolean arrays using **boolean operators**. In Python, these boolean operators are `and`, `or`, and `not`. But in pandas, there is a slight difference compared to Python. Take a look at the chart below: 
# MAGIC
# MAGIC |pandas|Python equivalent|Meaning|
# MAGIC |-|-|-|
# MAGIC |a & b| a and b| True if both a and b are True, else False|
# MAGIC | a \| b| a or b| True if either a or b is True|
# MAGIC |~a| not a | True if a is False, else False|
# MAGIC
# MAGIC Let's try to use the syntaxes in the table in the small example below:

# COMMAND ----------

cols = ["company", "revenues", "country"]
f500_sel = f500[cols].head()
f500_sel.head()

# COMMAND ----------

# MAGIC %md
# MAGIC We want to find the companies in `f500_sel` with more than 265 billion in revenue, and on top of that with headquarters located in China. We can achieve this by using two boolean comparisons like this:

# COMMAND ----------

over_265 = f500_sel["revenues"] > 265000
china = f500_sel["country"] == "China"
print(over_265.head())
print(china.head())

# COMMAND ----------

# MAGIC %md
# MAGIC What we can do next is to use the `&` operator to combine the two boolean arrays to get the actual boolean we want, like this:

# COMMAND ----------

combined = over_265 & china
combined.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Last but not least, we perform selection on our dataframe to get the final result like this:

# COMMAND ----------

final_cols = ["company", "revenues"]
result = f500_sel.loc[combined, final_cols]
result.head()

# COMMAND ----------

# MAGIC %md
# MAGIC This is the end result which fulfills all of our criteria.
# MAGIC
# MAGIC ### Task 3.5.7
# MAGIC Now try to do a similar task by yourself:
# MAGIC 1. Select all companies with revenues over **100 billion** and **negative profits** from the `f500` dataframe. Note that the entries in the profits column are given in millions of dollars (USD). The result should include all columns.
# MAGIC     - Create a boolean array that selects the companies with revenues greater than 100 billion. Assign the result to `large_revenue`.
# MAGIC     - Create a boolean array that selects the companies with profits less than `0`. Assign the result to `negative_profits`.
# MAGIC     - Combine `large_revenue` and `negative_profits`. Assign the result to `combined`.
# MAGIC     - Use combined to filter `f500`. Assign the result to `big_rev_neg_profit`.

# COMMAND ----------

# Start your code below:

large_revenue = f500["revenues"] > 100000
negative_profits = f500["profits"] < 0
combined = large_revenue & negative_profits
big_rev_neg_profit = f500[combined]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Sorting Values
# MAGIC
# MAGIC Now let's try to answer some more complicated questions about our data set. What if we wanted to find the company that employs the most people in China? How can we achieve this? We can first select all of the rows where the `country` column equals `China`, like this:

# COMMAND ----------

selected_rows = f500[f500["country"] == "China"]

# COMMAND ----------

# MAGIC %md
# MAGIC Then, we can use the [`DataFrame.sort_values()` method](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html) to sort the rows on the employees column, like this:

# COMMAND ----------

sorted_rows = selected_rows.sort_values("employees")
print(sorted_rows[["company", "country", "employees"]].head())

# COMMAND ----------

# MAGIC %md
# MAGIC The `sort_values()` method will by default automatically sort the rows in ascending order — from smallest to largest.
# MAGIC
# MAGIC But if we want to sort the rows in descending order instead, we can achieve this by setting the `ascending` parameter to `False`, like this:

# COMMAND ----------

sorted_rows = selected_rows.sort_values("employees", ascending=False)
print(sorted_rows[["company", "country", "employees"]].head())

# COMMAND ----------

# MAGIC %md
# MAGIC Now we see the Companies in China who employ the most people is China National Petroleum. 
# MAGIC
# MAGIC Can you find out the same about Japanese company?
# MAGIC ### Task 3.5.8
# MAGIC
# MAGIC 1. Find the companies headquartered in Japan with the largest number of employees.
# MAGIC     - Select only the rows that have a country name equal to `Japan`.
# MAGIC     - Use `DataFrame.sort_values()` to sort those rows by the `employees` column in descending order.
# MAGIC     - Use `DataFrame.iloc[]` to select the first row from the sorted dataframe.
# MAGIC     - Extract the company name from the index label `company` from the first row. Assign the result to `top_japanese_employer`.

# COMMAND ----------

# Start your code below:

japan = f500[f500["country"] == "Japan"]
sorted_rows = japan.sort_values("employees", ascending=False)
top_japanese_employer = sorted_rows.iloc[0,]
top_japanese_employer = top_japanese_employer.loc["company"]
