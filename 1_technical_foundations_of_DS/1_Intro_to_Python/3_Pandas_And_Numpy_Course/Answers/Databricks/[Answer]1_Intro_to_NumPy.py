# Databricks notebook source
# MAGIC %md
# MAGIC ## About This Notebook 
# MAGIC In this **Introduction to NumPy** chapter, we will learn:
# MAGIC - How vectorization makes our code faster.
# MAGIC - About n-dimensional arrays, and NumPy's ndarrays.
# MAGIC - How to select specific items, rows, columns, 1D slices, and 2D slices from ndarrays.
# MAGIC - How to apply simple calculations to entire ndarrays.
# MAGIC - How to use vectorized methods to perform calculations across any axis of an ndarray.
# MAGIC ***
# MAGIC ## 1. Introduction
# MAGIC
# MAGIC We have finished the fundamentals of Python programming in the previous two courses. In this course, we'll build on that knowledge to learn data analysis with some of the most powerful Python libraries for working with data.
# MAGIC
# MAGIC Have you ever wondered why the Python language is so popular? One straight forward answer is that Python makes writing programs easy. Python is a **high-level language**, which means we don’t need to worry about allocating memory or choosing how certain operations are done by our computers' processors like we have to when we use a **low-level language**, such as C. It takes usually more time to code in a low-level language; however, it also gives us more ability to optimize the code in order for it to run faster.
# MAGIC
# MAGIC We have two Python libraries that enable us to write code efficiently without sacrificing performance: <b>NumPy</b> and<b> pandas</b>.
# MAGIC
# MAGIC Now let's take a closer look at NumPy.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Introduction to Ndarrays
# MAGIC
# MAGIC The core data structure in NumPy is the <b>ndarray </b>or <b> n-dimensional array</b>. In data science,<b> array </b>describes a collection of elements, similar to a list. The word <b>n-dimensional </b>refers to the fact that ndarrays can have one or more dimensions. Let's first begin this session by working with one-dimensional (1D) ndarrays.
# MAGIC
# MAGIC In order to use the NumPy library, the first step is to import numpy into our Python environment like this:
# MAGIC
# MAGIC ````python
# MAGIC import numpy as np
# MAGIC ````
# MAGIC Note that ``np`` is the common alias for numpy.

# COMMAND ----------

# MAGIC %md
# MAGIC With the NumPy library a list can be directly converted to an ndarray using the `numpy.array()` [constructor](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.array.html).
# MAGIC How can we create a 1D ndarray? Look at the code below:
# MAGIC
# MAGIC ````python
# MAGIC data_ndarray = np. array([5,10,15,20])
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC Have you noticed that we used the syntax `np.array()` instead of `numpy.array()`? This is because we used the alias `as np` in our code: 
# MAGIC ````python
# MAGIC import numpy as np
# MAGIC ````
# MAGIC Now, let's do some exercises creating 1D ndarrays.
# MAGIC
# MAGIC ### Task 3.1.2:
# MAGIC 1. Import `numpy` and assign it to the alias `np`.
# MAGIC 2. Create a NumPy ndarray from the list `[10, 20, 30]`. Assign the result to the variable `data_ndarray`.

# COMMAND ----------

# Start your code below:

import numpy as np
data_ndarray = np.array([10,20,30])
print(data_ndarray)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. NYC Taxi-Airport Data
# MAGIC
# MAGIC So far we've only created one-dimensional ndarrys. However, ndarrays can also be two-dimensional. 
# MAGIC To illustrate this, we will analyze New York City taxi trip data released by the city of New York.
# MAGIC
# MAGIC Our dataset is stored in a [CSV file](https://en.wikipedia.org/wiki/Comma-separated_values) called <b>nyc_taxis.csv</b>. To convert the data set into a 2D ndarray, we'll first use Python's built-in csv [module](https://docs.python.org/3/library/csv.html) to import our CSV as a "list of lists". Then we can convert the lists of lists to an ndarray like this:

# COMMAND ----------

# MAGIC %md
# MAGIC Our list of lists is stored as `data_list`:
# MAGIC ````python
# MAGIC data_ndarray = np.array(data_list)
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC Below is the information about selected columns from the data set:
# MAGIC - `pickup_year`: The year of the trip.
# MAGIC - `pickup_month`: The month of the trip (January is 1, December is 12).
# MAGIC - `pickup_day`: The day of the month of the trip.
# MAGIC - `pickup_location_code`: The airport or borough where the trip started.
# MAGIC - `dropoff_location_code`: The airport or borough where the trip finished.
# MAGIC - `trip_distance`: The distance of the trip in miles.
# MAGIC - `trip_length`: The length of the trip in seconds.
# MAGIC - `fare_amount`: The base fare of the trip, in dollars.
# MAGIC - `total_amount`: The total amount charged to the passenger, including all fees, tolls and tips.
# MAGIC
# MAGIC ### Task 3.1.3 (IMPORTANT):
# MAGIC We have used Python's csv module to import the nyc_taxis.csv file and convert it to a list of lists containing float values.
# MAGIC
# MAGIC 1. Add a line of code using the `numpy.array()` constructor to convert the `converted_taxi_list` variable to a NumPy ndarray.
# MAGIC 2. Assign the result to the variable name `taxi`.

# COMMAND ----------

import csv
import numpy as np

# import nyc_taxi.csv as a list of lists
with open("../../../../../Data/nyc_taxis.csv", "r") as f:
    taxi_list = list(csv.reader(f))

# remove the header row
taxi_list = taxi_list[1:]

# convert all values to floats
converted_taxi_list = []
for row in taxi_list:
    converted_row = []
    for item in row:
        converted_row.append(float(item))
    converted_taxi_list.append(converted_row)

    
# Start your code below:

taxi = np.array(converted_taxi_list)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Array Shapes
# MAGIC
# MAGIC if we want, we can use the `print()` function to take a look at the data in the `taxi` variable.

# COMMAND ----------

# the code below only works if you have solved task 1.2
print(taxi)

# COMMAND ----------

# MAGIC %md
# MAGIC The elipses (...) between rows and columns indicate that there is more data in our NumPy ndarray than can easily be printed. In order to know the number of rows and columns in an ndarray, we can use the `ndarray.shape` attribute like this: 

# COMMAND ----------

import numpy as np
data_ndarray = np.array([[5, 10, 15], 
                         [20, 25, 30]])
print(data_ndarray.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC A data type **tuple** is returned as the result. Recall what we learned in the previous course about tuple — this type of value can't be modified.
# MAGIC
# MAGIC This value output gives us the following information:
# MAGIC 1. The first number tells us that there are 2 rows in `data_ndarray`.
# MAGIC 2. The second number tells us that there are 3 columns in `data_ndarray`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Selecting and Slicing Rows and Items from ndarrays
# MAGIC
# MAGIC The following code will compare working with ndarrays and list of lists to select one or more rows of data:

# COMMAND ----------

# MAGIC %md
# MAGIC #### List of lists method:
# MAGIC ````python
# MAGIC # Selecting a single row
# MAGIC sel_lol = data_lol[1]
# MAGIC
# MAGIC #Selecting multiple rows
# MAGIC sel_lol = data_lol[2:]
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC #### NumPy method:
# MAGIC
# MAGIC ````python
# MAGIC # Selecting a single row
# MAGIC sel_np = data_np[1]
# MAGIC
# MAGIC #Selecting multiple rows
# MAGIC sel_np = data_np[2:]
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC You see that the syntax of selecting rows in ndarrays is very similar to lists of lists. In fact, the syntax that we wrote above is a kind of shortcut. For any 2D array, the full syntax for selecting data is:

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC ndarray[row_index,column_index]
# MAGIC ````
# MAGIC
# MAGIC When you want to select the entire columns for a given set of rows, you just need to do this:
# MAGIC ````python
# MAGIC ndarray[row_index]
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC Here `row_index` defines the location along the row axis and `column_index` defines the location along the column axis.
# MAGIC
# MAGIC Like lists, array slicing is from the first specified index up to — but **not including** – the second specified index. For example, to select the items at index 1, 2, and 3, we'd need to use the slice `[1:4]`.
# MAGIC
# MAGIC This is how we **select a single item** from a 2D ndarray:

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### List of lists method
# MAGIC
# MAGIC ````python
# MAGIC # Selecting a single row
# MAGIC sel_lol = data_lol[1][3]
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC #### NumPy method
# MAGIC
# MAGIC ````python
# MAGIC # Selecting a single row
# MAGIC sel_np = data_np[1,3] # The comma here separates row/column locations. Produces a single Python object.
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC Two separate pairs of square brackets back-to-back are used with a list of lists and a single pair of brackets with comma-separated row and column locations is used with a NumPy ndarray.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.1.5:
# MAGIC From the `taxi` ndarray:
# MAGIC
# MAGIC 1. Select the row at index `0`. Assign it to `row_0`.
# MAGIC 2. Select every column for the rows from index `391` up to and including `500`. Assign them to `rows_391_to_500`.
# MAGIC 3. Select the item at row index `21` and column index `5`. Assign it to `row_21_column_5`.

# COMMAND ----------

# Start your code below:

row_0 = taxi[0]
rows_391_to_500 = taxi[391:501]
row_21_column_5 = taxi[21,5]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Selecting Columns and Custom Slicing ndarrays
# MAGIC
# MAGIC Let's take a look at how to select one or more columns of data:

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### List of lists method
# MAGIC ````python
# MAGIC # Selecting a single row
# MAGIC sel_lol = []
# MAGIC
# MAGIC for row in data_lol:
# MAGIC     col4 = row[3]
# MAGIC     sel_lol.append(col4)
# MAGIC     
# MAGIC #Selecting multiple columns
# MAGIC sel_lol = []
# MAGIC
# MAGIC for row in data_lol:
# MAGIC     col23 = row[2:3]
# MAGIC     sel_lol.append(col23)
# MAGIC     
# MAGIC #Selecting multiple, specific columns
# MAGIC sel_lol = []
# MAGIC
# MAGIC for row in data_lol:
# MAGIC     cols = [row[1], row[3], row[4]]
# MAGIC     sel_lol.append(cols)
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC #### NumPy Method
# MAGIC
# MAGIC ````python
# MAGIC # Selecting a single row
# MAGIC sel_np = data_np[:,3] #Produces a 1D ndarray
# MAGIC     
# MAGIC #Selecting multiple columns
# MAGIC sel_np = data_np[:, 1:3] # Produces a 2D ndarray
# MAGIC     
# MAGIC #Selecting multiple, specific columns
# MAGIC cols = [1, 3, 4]
# MAGIC sel_np = data_np[:,cols] # Produces a 2D ndarray``
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC You see that with a list of lists, we need to use a for loop to extract specific column(s) and append them back to a new list. It is much easier with ndarrays. We again use single brackets with comma-separated row and column locations, but we use a colon (`:`) for the row locations, which gives us all of the rows.
# MAGIC
# MAGIC If we want to select a partial 1D slice of a row or column, we can combine a single value for one dimension with a slice for the other dimension:

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### List of lists method
# MAGIC ````python
# MAGIC # Selecting a 1D slice (row)
# MAGIC sel_lol = data_lol[2][1:4]  #third row (row index of 2) of column 1, 2, 3
# MAGIC
# MAGIC #Selecting a 1D slice (column)
# MAGIC sel_lol = []
# MAGIC
# MAGIC rows = data_lol[1:5] #fifth column (column index of 4) of row 1, 2, 3, 4
# MAGIC for r in rows:
# MAGIC     col5 = r[4]
# MAGIC     sel_lol.append(col5) 
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC #### NumPy Method
# MAGIC
# MAGIC ````python
# MAGIC # Selecting a 1D slice (row)
# MAGIC sel_np = data_np[2, 1:4] # Produces a 1D ndarray
# MAGIC     
# MAGIC # Selecting a 1D slice (column)
# MAGIC sel_np = data_np[1:5, 4] # Produces a 1D ndarray
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC Lastly, if we want to select a 2D slice, we can use slices for both dimensions:

# COMMAND ----------

# MAGIC %md
# MAGIC #### List of lists method
# MAGIC
# MAGIC ````python
# MAGIC # Selecting a 2D slice 
# MAGIC sel_lol = []
# MAGIC
# MAGIC rows = data_lol[1:4]
# MAGIC for r in rows:
# MAGIC     new_row = r[:3]
# MAGIC     sel_lol.append(new_row)
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC #### NumPy method
# MAGIC
# MAGIC ````python
# MAGIC # Selecting a 2D slice 
# MAGIC sel_np = data_np[1:4,:3]
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.1.6:
# MAGIC From the `taxi` ndarray:
# MAGIC
# MAGIC 1. Select every row for the columns at indexes `1`, `4`, and `7`. Assign them to `columns_1_4_7`.
# MAGIC 2. Select the columns at indexes `5` to `8` inclusive for the row at index `99`. Assign them to `row_99_columns_5_to_8`.
# MAGIC 3. Select the rows at indexes `100` to `200` inclusive for the column at index `14`. Assign them to `rows_100_to_200_column_14`.

# COMMAND ----------

# Start your code below:

columns_1_4_7 = taxi[:,[1,4,7]]
row_99_columns_5_to_8 = taxi[99,5:9]
rows_100_to_200_column_14 = taxi[100:201,14]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Vector Math
# MAGIC In this section we will explore the power of vectorization. Take a look at the example below:

# COMMAND ----------

# convert the list of lists to an ndarray
my_numbers = [[1,2,3],[4,5,6], [7,8,9]] 
my_numbers = np.array(my_numbers)

# select each of the columns - the result
# of each will be a 1D ndarray
col1 = my_numbers[:,0]
col2 = my_numbers[:,1]

# add the two columns
sums = col1 + col2
sums

# COMMAND ----------

# MAGIC %md
# MAGIC The code above can be simplified into one line of code, like this:

# COMMAND ----------

sums = my_numbers[:,0] + my_numbers[:,1]
sums

# COMMAND ----------

# MAGIC %md
# MAGIC Some key take aways from the code above:
# MAGIC - When we selected each column, we used the syntax `ndarray[:,c]` where `c` is the column index we wanted to select. Like we saw in the previous screen, the colon selects all rows.
# MAGIC - To add the two 1D ndarrays, `col1` and `col2` we can simply put the addition operator ``+`` between them.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Vector Math Continued
# MAGIC
# MAGIC Do you know that the standard Python numeric operators also work with vectors such as:
# MAGIC
# MAGIC - **Addition**: `vector_a + vector_b`
# MAGIC - **Subtraction**: `vector_a - vector_b`
# MAGIC - **Multiplication**: (unrelated to the vector multiplication in linear algebra): `vector_a * vector_b`
# MAGIC - **Division**: `vecotr_a / vector_b`
# MAGIC
# MAGIC Note that all these operations are entry-wise.
# MAGIC
# MAGIC Below is an example table from our taxi data set:

# COMMAND ----------

# MAGIC %md
# MAGIC |trip_distance|trip_length|
# MAGIC |-------------|-----------|
# MAGIC |21.00|2037.0|
# MAGIC |16.29|1520.0|
# MAGIC |12.70|1462.0|
# MAGIC |8.70|1210.0|
# MAGIC |5.56|759.0|

# COMMAND ----------

# MAGIC %md
# MAGIC We want to use these columns to calculate the average travel speed of each trip in miles per hour. For this we can use the formula below: <br>
# MAGIC **miles per hour = distance in miles / length in hours**
# MAGIC
# MAGIC The current column `trip_distance` is already expressed in miles, but `trip_length` is expressed in seconds. First, we want to convert `trip_length` into hours:

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC trip_distance = taxi[:,7]
# MAGIC trip_length_seconds = taxi[:,8]
# MAGIC
# MAGIC trip_length_hours = trip_length_seconds / 3600 
# MAGIC ````
# MAGIC Note: 3600 seconds is one hour

# COMMAND ----------

# MAGIC %md
# MAGIC We can then divide each value in the vector by a single number, 3600, instead of another vector. Let's see the first five rows of the result below:
# MAGIC
# MAGIC |trip_length_hours|
# MAGIC |-------------|
# MAGIC |0.565833|
# MAGIC |0.422222|
# MAGIC |0.406111|
# MAGIC |0.336111|
# MAGIC |0.210833|

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Calculating Statistics For 1D ndarrays
# MAGIC
# MAGIC We've created ``trip_mph`` in the previous exercise. This is a 1D ndarray of the average mile-per-hour speed of each trip in our dataset. Now, something else we can do is to calculate the ``minimum``, ``maximum``, and ``mean`` values for `trip_distance`.
# MAGIC
# MAGIC In order to calculate the minimum value of a 1D ndarray, all we need to do is to use the vectorized `ndarray.min()` [method](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.min.html), like this:
# MAGIC
# MAGIC ````python
# MAGIC distance_min = trip_distance.min()
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC For other Numpy ndarrays methods we have:
# MAGIC - [ndarray.min()](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.min.html#numpy.ndarray.min) to calculate the minimum value
# MAGIC - [ndarray.max()](https://docs.scipy.org/doc/numpy-1.16.1/reference/generated/numpy.ndarray.max.html#numpy.ndarray.max) to calculate the maximum value
# MAGIC - [ndarray.mean()](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.mean.html#numpy.ndarray.mean) to calculate the mean or average value
# MAGIC - [ndarray.sum()](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.sum.html#numpy.ndarray.sum) to calculate the sum of the values
# MAGIC
# MAGIC You will find the full list of ndarray methods in the NumPy ndarray documentation [here](https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#calculation).
# MAGIC
# MAGIC ### Task 3.1.9:
# MAGIC 1. Use the `ndarray.max()` method to calculate the maximum value of `trip_distance`. Assign the result to `distance_max`.
# MAGIC 2. Use the `ndarray.mean()` method to calculate the average value of `trip_distance`. Assign the result to `distance_mean`.

# COMMAND ----------

# Selecting only the relevant column distance
trip_distance = taxi[:,7]
trip_distance_miles = taxi[:,7]
trip_length_seconds = taxi[:,8]

trip_length_hours = trip_length_seconds / 3600 # 3600 seconds is one hour
trip_mph = trip_distance_miles / trip_length_hours
# Start your code below:
mph_min = trip_mph.min()
mph_max = trip_mph.max()
mph_mean = trip_mph.mean()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Calculating Statistics For 1D ndarrays Continued (IMPORTANT)
# MAGIC
# MAGIC Let's examine the difference between methods and functions.
# MAGIC <b>Functions</b> act as stand alone segments of code that usually take an input, perform some processing, and return some output. Take for example the `len()` function used to calculate the length of a list or the number of characters in a string:

# COMMAND ----------

my_list = [25,18,9]
print(len(my_list))

# COMMAND ----------

my_string = 'RBI'
print(len(my_string))

# COMMAND ----------

# MAGIC %md
# MAGIC In contrast, <b>methods</b> are special functions that belong to a specific type of object. In other words, when we work with list objects, there are special functions or methods that can only be used with lists. For example, `list.append()` method is used to add an item to the end of a list. We will get an error if we use this method on a string:

# COMMAND ----------

my_string.append(' is the best!')

# COMMAND ----------

# MAGIC %md
# MAGIC There are cases in NumPy where operations are sometimes implemented as both methods and functions. It can be confusing at first glance, so let's take a look at an example.

# COMMAND ----------

# MAGIC %md
# MAGIC |Calculation|Function Representation|Method Representation|
# MAGIC |-------------|-----------|---------------|
# MAGIC |Calculate the minimum value of trip_mph|np.min(trip_mph)|trip_mph.min()|
# MAGIC |Calculate the maximum value of trip_mph|np.ax(trip_mph)|trip_mph.max()|
# MAGIC |Calculate the mean average value of trip_mph|np.mean(trip_mph)|trip_mph.mean()|
# MAGIC |Calculate the median average value of trip_mph|np.median(trip_mph)|There is no ndarray median method|
# MAGIC
# MAGIC To help you remember, you can see it as this:
# MAGIC - anything that starts with `np` (e.g. `np.mean()`) is a function 
# MAGIC - anything expressed with an object (or variable) name first (e.g. `trip_mph.mean()`) is a method
# MAGIC - it's up to you to decide which one to use
# MAGIC - however, it is more common to use the method approach 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Calculating Statistics For 2D ndarrays
# MAGIC
# MAGIC We have only worked with statistics for 1D ndarrays so far. If we use the `ndarray.max()` method on a 2D ndarray without any additional parameters, a single value will be returned, just like with a 1D array.
# MAGIC
# MAGIC What happens if we want to find the **maximum value of each row**?
# MAGIC Specification of the axis parameter is needed as an indication that we want to calculate the maximum value for each row.
# MAGIC
# MAGIC If we want to find the maximum value of each column, we'd use an axis value of 0 like this:
# MAGIC
# MAGIC ndarray.max(axis = 0)
# MAGIC
# MAGIC Let's use what we've learned to check the data in our taxi data set. Below is an example table of our data set:
# MAGIC
# MAGIC |fare_amount|fees_amount|tolls_amount|tip_amount|total_amount|
# MAGIC |-------------|-----------|--------|-------------|-----------|
# MAGIC |52.0|0.8|5.54|11.65|69.99|
# MAGIC |45.0|1.3|0.00|8.00|54.3|
# MAGIC |36.5|1.3|0.00|0.00|37.8|
# MAGIC |26.0|1.3|0.00|5.46|32.76|
# MAGIC |17.5|1.3|0.00|0.00|18.8|
# MAGIC
# MAGIC You see that **total amount = fare amount + fees amount + tolls amount + tip amount**.
# MAGIC
# MAGIC Now let's see if you can perform a 2D ndarray calculation on the data set.
# MAGIC
# MAGIC ### Task 3.1.11:
# MAGIC 1. Use the `ndarray.sum()` method to calculate the sum of each row in `fare_components`. Assign the result to `fare_sums`.
# MAGIC 2. Extract the 14th column in `taxi_first_five`. Assign to `fare_totals`.
# MAGIC 3. Print `fare_totals` and `fare_sums`. You should see the same numbers.

# COMMAND ----------

# get the table from above (first five rows of taxi and columns fare_amount, fees_amount, tolls_amount, tip_amount)
fare_components = taxi[:5,[9,10,11,12]]
# we'll compare against the first 5 rows only
taxi_first_five = taxi[:5]

# Start your code below:

# select these columns: fare_amount, fees_amount, tolls_amount, tip_amount
fare_components = taxi_first_five[:,9:13]
fare_sums = fare_components.sum(axis = 1)
fare_totals = taxi_first_five[:,13]
print(fare_totals)
print(fare_sums)
