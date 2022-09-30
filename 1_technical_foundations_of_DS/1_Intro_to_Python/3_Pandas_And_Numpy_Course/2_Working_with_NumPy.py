# Databricks notebook source
# MAGIC %md
# MAGIC ## About This Notebook 
# MAGIC In this **Working with NumPy** notebook, we will learn:
# MAGIC - How to use `numpy.genfromtxt()` to read in an ndarray.
# MAGIC - What a boolean array is, and how to create one.
# MAGIC - How to use boolean indexing to filter values in one and two-dimensional ndarrays.
# MAGIC - How to assign one or more new values to an ndarray based on their locations.
# MAGIC - How to assign one or more new values to an ndarray based on their values.
# MAGIC ***
# MAGIC ## 1. Reading CSV files with NumPy
# MAGIC 
# MAGIC In this chapter, we will learn a technique called <b>Boolean Indexing</b>. Before we dig deeper into this topic, let's first learn how to read files into NumPy ndarrays. Below is the simplified syntax of the function, as well as an explanation for the two parameters:
# MAGIC 
# MAGIC ````python
# MAGIC np.genfromtxt(filename, delimiter=None)
# MAGIC ````
# MAGIC 
# MAGIC - ``filename``: A positional argument, usually a string representing the path to the text file to be read.
# MAGIC - ``delimiter``: A named argument, specifying the string used to separate each value.
# MAGIC 
# MAGIC In our case, the data is stored in a CSV file, therefore the delimiter is a comma ",".
# MAGIC So this is how we can read in a file named ``data.csv``:
# MAGIC 
# MAGIC 
# MAGIC ````python
# MAGIC data = np.genfromtxt('data.csv', delimiter = ',')
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.2.1:
# MAGIC Now let's try to read our nyc_taxis.csv file into NumPy.
# MAGIC 
# MAGIC 1. Import the NumPy library and assign to the alias ``np``.
# MAGIC 2. Use the `np.genfromtxt()` function to read the nyc_taxis.csv file into NumPy. Assign the result to taxi. Do not forget to use also delimiter argument, such as shown above.
# MAGIC 3. Use the ``ndarray.shape`` attribute to assign the shape of taxi to ``taxi_shape``.

# COMMAND ----------

# Start your code here:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Reading CSV files with NumPy Continued
# MAGIC 
# MAGIC We have used the `numpy.genfromtxt()` function to read the ``nyc_taxis.csv`` file into NumPy in the previous notebook.
# MAGIC 
# MAGIC Just to refresh your memory, in the previous mission we have done something like this:

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC # import nyc_taxi.csv as a list of lists
# MAGIC f = open("nyc_taxis.csv", "r")
# MAGIC taxi_list = list(csv.reader(f))
# MAGIC 
# MAGIC # remove the header row
# MAGIC taxi_list = taxi_list[1:]
# MAGIC 
# MAGIC # convert all values to floats
# MAGIC converted_taxi_list = []
# MAGIC for row in taxi_list:
# MAGIC     converted_row = []
# MAGIC     for item in row:
# MAGIC         converted_row.append(float(item))
# MAGIC     converted_taxi_list.append(converted_row)
# MAGIC 
# MAGIC taxi = np.array(converted_taxi_list)
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC Have you noticed that we converted all the values to floats before we converted the list of lists to an ndarray? 
# MAGIC > The reason for this is because that NumPy ndarrays can contain only **one datatype**.
# MAGIC 
# MAGIC This part of the code was omitted in the previous exercise, because when `numpy.getfromtxt()` is called, the function automatically tries to determine the data type of the file by looking at the values.
# MAGIC 
# MAGIC To see which datatype we have in the ndarray, simply use `ndarray.dtype` attribute like this:
# MAGIC ````python
# MAGIC print(taxi.dtype)
# MAGIC ````
# MAGIC ### Task 3.2.2:
# MAGIC 1. Use the `numpy.genfromtxt()` function to again read the nyc_taxis.csv file into NumPy, but this time, skip the first row. Assign the result to `taxi`.
# MAGIC 2. Assign the shape of `taxi` to `taxi_shape`.

# COMMAND ----------

# Start your code here:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Boolean Arrays
# MAGIC 
# MAGIC In this session, we're going to focus on the boolean array.
# MAGIC 
# MAGIC Do you remember that the boolean (or bool) type is a built-in Python type that can be one of two unique values:
# MAGIC 
# MAGIC - True
# MAGIC - False
# MAGIC 
# MAGIC Do you also remember that we've used boolean values when working with Python comparison operators like 
# MAGIC - ``==`` equal
# MAGIC - ``>`` greater than
# MAGIC - ``<`` less than
# MAGIC - ``!=`` not equal
# MAGIC 
# MAGIC See a couple examples of simple boolean operations below just to refresh your memory:

# COMMAND ----------

print(type(3.5) == float)

# COMMAND ----------

print(5 > 6)

# COMMAND ----------

# MAGIC %md
# MAGIC In the previous notebook where we explored vector operations we learned that the result of an operation between a ndarray and a single value is a new ndarray:

# COMMAND ----------

import numpy as np
print(np.array([2,4,6,8]) + 10)

#The + 10 operation is applied to each value in the array

# COMMAND ----------

# MAGIC %md
# MAGIC Guess what happens when we perform a **boolean operation** between an ndarray and a single value:

# COMMAND ----------

import numpy as np
print(np.array([2,4,6,8]) < 5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Boolean Indexing with 1D ndarrays
# MAGIC 
# MAGIC In the last exercise, we learned how to create boolean arrays using vectorized boolean operations. Now, I want to show you a technique known as **boolean indexing**, (or index/select) using boolean arrays.
# MAGIC See an example from the previous notebook:

# COMMAND ----------

c = np.array([80.0, 103.4, 96.9, 200.3])
c_bool = c > 100
print(c_bool)

# COMMAND ----------

# MAGIC %md
# MAGIC How do we index using our new boolean array? All we need to do is to use the square brackets like this:

# COMMAND ----------

result = c[c_bool]
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC The boolean array acts as a filter, the values that corresponding to **True** become part of the result and the values that corresponding to **False** are removed from the final list.
# MAGIC 
# MAGIC How can we use boolean indexing knowledge in our data set?
# MAGIC For example, to confirm the number of taxi rides from the month of january, we can do this:

# COMMAND ----------

# First, select just the pickup_month column (second column in the ndarray with column index 1)
pickup_month = taxi[:,1]

# use a boolean operation to make a boolean array, where the value 1 corresponds to January
january_bool = pickup_month == 1

# use the new boolean array to select only the items from pickup_month that have a value of 1
january = pickup_month[january_bool]

# use the .shape attribute to find out how many items are in our january ndarray
january_rides = january.shape[0]
print(january_rides)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.2.4:
# MAGIC 
# MAGIC 1. Calculate the number of rides in the taxi ndarray that are from **February**:
# MAGIC     - Create a boolean array, ``february_bool``, that evaluates whether the items in ``pickup_month`` are equal to ``2``.
# MAGIC     - Use the ``february_bool`` boolean array to index ``pickup_month``. Assign the result to ``february``.
# MAGIC     - Use the ``ndarray.shape`` attribute to find the number of items in `february`. Assign the result to ``february_rides``.

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Boolean Indexing with 2D ndaarays
# MAGIC 
# MAGIC Now it is time to use boolean indexing with ``2D ndarrays``. 
# MAGIC > One thing to keep in mind is that the boolean array must have the same length as the dimension you're indexing. This is one of the constraints when we work with 2D ndarrays.

# COMMAND ----------

arr = np.array([
                [1,2,3],
                [4,5,6],
                [7,8,9],
                [10,11,12]
])

print(arr)

# COMMAND ----------

bool_1 = [True, False, 
        True, True]
print(arr[bool_1])

# COMMAND ----------

print(arr[:, bool_1])

# COMMAND ----------

# MAGIC %md
# MAGIC You see that `bool_1`'s shape (4) is not the same as the shape of `arr`'s second axis(3), so it can't be used to index and produces an error.

# COMMAND ----------

bool_2 = [False, True, True]
print(arr[:,bool_2])

# COMMAND ----------

# MAGIC %md
# MAGIC `bool_2`'s shape (3) is the same as the shape of `arr`'s second axis (3), so this selects the 2nd and 3rd columns.

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's apply what we have learned to our data set. This time we will analyze the average speed of trips. Recall that we calculated the ``average travel speed `` as follows:
# MAGIC ````python
# MAGIC trip_mph = taxi[:,7] / (taxi[:,8] / 3600)
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC Next, how do we check for trips with an average speed greater than 20,000 mph?

# COMMAND ----------

trip_mph = taxi[:,7] / (taxi[:,8] / 3600)

# COMMAND ----------

# create a boolean array for trips with average
# speeds greater than 20,000 mph
trip_mph_bool = trip_mph > 20000

# use the boolean array to select the rows for
# those trips, and the pickup_location_code,
# dropoff_location_code, trip_distance, and
# trip_length columns
trips_over_20000_mph = taxi[trip_mph_bool,5:9]

print(trips_over_20000_mph)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.2.5 (HARD):
# MAGIC 1. Create a boolean array, ``tip_bool``, that determines which rows have values for the `tip_amount` column of more than 50.<br>
# MAGIC Hint: You might have to examine the original nyc_taxis.csv file to find an index of desired column.
# MAGIC 2. Use the ``tip_bool`` array to select all rows from taxi with values tip amounts of more than 50, and the columns from indexes `5` to `13` inclusive. Assign the resulting array to ``top_tips``.

# COMMAND ----------

# Start your code below


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Assigning Values in ndarrays (OPTIONAL)
# MAGIC 
# MAGIC After having learned how to retrieve data from ndarrays, now we will use the same indexing techniques to modify values within an ndarray. The syntax looks like this: <br>
# MAGIC 
# MAGIC ````python
# MAGIC ndarray[location_of_values] = new_value
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC With 1D array, all we need to do is to specify one specific index location like this:

# COMMAND ----------

a = np.array(['red','blue','black','blue','purple'])
a[0] = 'orange'
print(a)

# COMMAND ----------

# MAGIC %md
# MAGIC Or multiple values can be assigned at once:

# COMMAND ----------

a[3:] = 'pink'
print(a)

# COMMAND ----------

# MAGIC %md
# MAGIC With a 2D ndarray, just like with a 1D ndarray, we can assign one specific index location:

# COMMAND ----------

ones = np.array([[1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1]])
ones[1,2] = 99
print(ones)

# COMMAND ----------

# MAGIC %md
# MAGIC Or we can assign a whole row:

# COMMAND ----------

ones[0] = 42
print(ones)

# COMMAND ----------

# MAGIC %md
# MAGIC Or a whole column:

# COMMAND ----------

ones[:,2] = 0
print(ones)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Assignment Using Boolean Arrays (OPTIONAL)
# MAGIC 
# MAGIC Boolean arrays become extremely powerful when used for assignment, like this:

# COMMAND ----------

a2 = np.array([1, 2, 3, 4, 5])

a2_bool = a2 > 2

a2[a2_bool] = 99

print(a2)

# COMMAND ----------

# MAGIC %md
# MAGIC The boolean array has the ability to control the values that the assignment applies to, and the other values remain unchanged.

# COMMAND ----------

a = np.array([1, 2, 3, 4, 5])

a [ a > 2] = 99

print(a)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Assignment Using Boolean Arrays Continued (OPTIONAL)
# MAGIC 
# MAGIC Now let's take a look at an example of assignment using a boolean array with two dimensions:

# COMMAND ----------

b = np.array([
                [1,2,3],
                [4,5,6],
                [7,8,9]           
])

b[b > 4] = 99
print(b)

# The b > 4 boolean operation produces a 2D boolean array 
# which then controls the values that the assignment applies to.

# COMMAND ----------

# MAGIC %md
# MAGIC We can also use a 1D boolean array to perform assignment on a 2D array:

# COMMAND ----------

c = np.array([
                [1,2,3],
                [4,5,6],
                [7,8,9]           
])

c[c[:,1] > 2, 1] = 99

print(c)

# COMMAND ----------

# MAGIC %md
# MAGIC The above code selected the second column (with column index 1), and used boolean index technique (which value is > 2). The boolean array is only applied to the second column, while all other values remaining unchanged.
# MAGIC 
# MAGIC The pseudocode syntax for this code is the following, first we used an intermediate variable:

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC bool = array[:, column_for_comparison] == value_for_comparison
# MAGIC array[bool, column_for_assignment] = new_value
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC And now all in one line:

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC array[array[:, column_for_comparison] == value_for_comparison, column_for_assignment] = new_value
# MAGIC ````
