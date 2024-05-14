# Databricks notebook source
# MAGIC %md
# MAGIC # Transforming the data

# COMMAND ----------

# importing essential packages
import pandas as pd
import numpy as np
import math

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. [`.map()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html) method
# MAGIC
# MAGIC - in Pandas the [`.map()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html) method is a great tool to have when it comes to processing & transforming a Series
# MAGIC - it is a convenient way to perform _element-wise_ transformations and other data cleaning-related operations
# MAGIC - this method on Series takes a function object or an input mapping as argument
# MAGIC - any function that takes a single argument and returns a value can be used with [`.map()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 1.1 - Example using For Loops

# COMMAND ----------

# Run this code
our_series = pd.Series(['This', 'is', 'the', 'first', 'example'])

# COMMAND ----------

#Step 1: intialize an empty list "result_loop" that will store our results later
#Step 2: get the length of each variable in the list "our_list"
#Step 3: append the result to the list "result_loop"
#Step 4: Convert to a Pandas Series and print the result

result_loop = []

for word in our_series:
    result_loop.append(len(word))

result_loop = pd.Series(result_loop)

print(result_loop)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 1.2 - Example using [`.map()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html) method

# COMMAND ----------

# Run this code
our_series = ['This', 'is', 'the', 'first', 'example']

# COMMAND ----------

# Step 1: Use the .map() method to get the length of the words in our_list
# Step 2: Assign the result to the variable name "result_map" to print it to the screen

result_map = our_series.map(len)
print(result_map)

# COMMAND ----------

# MAGIC %md
# MAGIC In the above example the [`.map()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html) method iterates over `our_list`, applies the function on each element and returns the length of the strings as a new list.

# COMMAND ----------

# MAGIC %md
# MAGIC Which one do you think is neater and shorter?
# MAGIC
# MAGIC ```python
# MAGIC result_loop = []
# MAGIC
# MAGIC for word in our_series:
# MAGIC     result_loop.append(len(word))
# MAGIC
# MAGIC result_loop = pd.Series(result_loop)

# MAGIC
# MAGIC print(result_loop)
# MAGIC ```
# MAGIC vs. 
# MAGIC
# MAGIC ```python
# MAGIC result_map = our_series.map(len)
# MAGIC print(result_map)
# MAGIC ```
# MAGIC
# MAGIC In the programming world, it is cleaner and much more concise and sophisticated to use [`.map()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html) instead of for-loops. On top of that, with [`.map()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html) you can guarantee that the original sequence won't be acccidentally mutated or changed, since [`.map()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html) always returns a sequence of the results and leads to fewer errors in code. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 1.3 - Using [`.map()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html) with a dictionary
# MAGIC
# MAGIC For a Pandas Series the [`.map()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html) method can also be used with a dictionary instead of a function.
# MAGIC This is especially usefull if each value is simply replaced by another.  
# MAGIC

# COMMAND ----------

# run this code
animal_series = pd.Series(["wolf", "iguana", "shark", "capybara"])

# COMMAND ----------

# Step 1: Create a dictionary mapping the values from the Series to the desired replacement values.
# Step 2: Use the .map() method to replace the values in animal_series
# Step 3: Assign the result to the variable name "class_series" to print it to the screen

class_dict = {"wolf": "mammal", "capybara": "mammal", "iguana": "reptile", "shark": "fish"}

class_series = animal_series.map(class_dict)
class_series

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bulit-in [``map()``](https://docs.python.org/3/library/functions.html#map) function
# MAGIC
# MAGIC In addition to the Pandas Series method there is also a build in [``map()``](https://docs.python.org/3/library/functions.html#map) function.
# MAGIC This funciton works on any iterable object in Python (e.g. list, tuple, dictionary, set, or Series).
# MAGIC
# MAGIC - it returns an iterator (don't worry about this concept for now)
# MAGIC - the resulting values (an iterator) can be passed to the [`list()`](https://docs.python.org/3/library/functions.html#func-list) function or [`set()`](https://docs.python.org/3/library/functions.html#func-set) function to create a list or a set
# MAGIC
# MAGIC Example code:
# MAGIC
# MAGIC `map(function, iterable)`
# MAGIC
# MAGIC To extract the result we can use for example: <break> 
# MAGIC
# MAGIC `list(map(function, iterable))`
# MAGIC
# MAGIC or 
# MAGIC
# MAGIC `set(map(function, iterable))`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 1.4 - Using built-in [``map()``](https://docs.python.org/3/library/functions.html#map) function

# COMMAND ----------

# run this code
our_list = ['This', 'is', 'the', 'first', 'example']

# COMMAND ----------

# Step 1: Use the .map() method to get the length of the words in our_list
# Step 2: Assign the result to the variable name "result_map" to print it to the screen

result_map = list(map(len, our_list))
result_map

# COMMAND ----------

# MAGIC %md

# MAGIC Feel free to check out [this](https://stackoverflow.com/questions/1975250/when-should-i-use-a-map-instead-of-a-for-loop#:~:text=4%20Answers&text=map%20is%20useful%20when%20you,loop%20and%20constructing%20a%20list.) on stackoverflow, where the advantages of using map over for-loops are discussed.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1
# MAGIC Now, let's create a function `triple` and a Pandas Series `numbers` which will be our iterable.

# COMMAND ----------

# Run this code 
def triple(x):
    return x * 3

# COMMAND ----------

# Run this code
numbers = pd.Series([15, 4, 8, 45, 36, 7])

# COMMAND ----------

# TASK 1 >>>> Apply the .map() method with the function triple on our Pandas Series 'numbers' and store it in the variable result_2 
#             Print result_2 (the result should be the numbers multiply by 3)
#             Think about the three different steps performed in Example 1

### Start your code below ###


# COMMAND ----------

# MAGIC %md
# MAGIC # 2. [``filter()``](https://docs.python.org/3/library/functions.html#filter) function
# MAGIC
# MAGIC - similar to the built-in [`map()`](https://docs.python.org/3/library/functions.html#map) function, but instead of any function, [`filter()`](https://docs.python.org/3/library/functions.html#filter) takes a Boolean-valued function (a function that returns True or False based on the input data) instead of any built-in functions and a sequence of iterables (list, tuple, dictionary, set, or Series) as arugments

# MAGIC - returns the items of the intput data which the Boolean-valued function returns `True`
# MAGIC - the Boolean-valued function can be used-defined function

# COMMAND ----------

# MAGIC %md
# MAGIC Imagine there is a list with positive and negative numbers

# COMMAND ----------

# Run this code
list_mixed = [-1,0,2,24,-42,-5,30,99]

# COMMAND ----------

# Run this code
def criteria(x): 
    return x >= 0

# COMMAND ----------

# MAGIC %md
# MAGIC With the help of filter and our own user-defined function we can filter out the negative values and be left with only positive values.

# COMMAND ----------

list_positive = list(filter(criteria, list_mixed))
print(list_positive)

# COMMAND ----------

# MAGIC %md
# MAGIC Did you know we can combine the [`map()`](https://docs.python.org/3/library/functions.html#map) and [`filter()`](https://docs.python.org/3/library/functions.html#filter) functions? Since [`filter()`](https://docs.python.org/3/library/functions.html#filter) returns a selected iterable based on certain criteria, the output of [`filter()`](https://docs.python.org/3/library/functions.html#filter) can be our input for the [`map()`](https://docs.python.org/3/library/functions.html#map) method.
# MAGIC
# MAGIC In order to avoid a negative number as an argument for [`math.sqrt()`](https://docs.python.org/3/library/math.html#math.sqrt) which will result in a `ValueError`, we want to filter out the negative numbers before we apply the [`math.sqrt()`](https://docs.python.org/3/library/math.html#math.sqrt) method.

# COMMAND ----------

list_sqrt = list(map(math.sqrt, filter(criteria, list_mixed)))
print(list_sqrt)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional: Task 2

# COMMAND ----------

# TASK 2 >>>> With the help of .map() and .filter(),
#             round up any number that is bigger than 5 from the list "list_sqrt" to the next whole digit.
#             To round up the number, you can use round().
#             Don't forget to write your user-defined function as your criteria to filter out the "not desirable" numbers

### Start your code below ###

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. [`.apply()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html) method
# MAGIC
# MAGIC - this method applies a function along an axis of the DataFrame \\(^{1}\\) or a Series
# MAGIC - it also works elementwise but is suited to more complex functions and operations
# MAGIC - it accepts user-defined functions which apply a transformation/aggregation on a DataFrame (or Series) as well
# MAGIC
# MAGIC You can find a nice comparison of [`.map()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.map.html) and [`.apply()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html) methods and when to use them in [this article on stackoverflow](https://stackoverflow.com/questions/19798153/difference-between-map-applymap-and-apply-methods-in-pandas).


# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 3.1

# COMMAND ----------

# Run this code
students = [(1, 'Robert', 30, 'Slovakia', 26),
           (2, 'Jana', 29, 'Sweden' , 27),
           (3, 'Martin', 31, 'Sweden', 26),
           (4, 'Kristina', 26,'Germany' , 30),
           (5, 'Peter', 33, 'Austria' , 22),
           (6, 'Nikola', 25, 'USA', 23),
           (7, 'Renato', 35, 'Brazil', 26)]

students_1 = pd.DataFrame(students, columns= ['student_id', 'first_name', 'age', 'country', 'score'])
print(students_1)

# COMMAND ----------

# Run this code to create a regular function

def score_func(x): 
    if x < 25: 
        return "Retake" 
    else: 
        return "Pass"

# COMMAND ----------

# Use .apply() along with score_func  
students_1['result'] = students_1.score.apply(score_func)
print(students_1)

# COMMAND ----------

# MAGIC %md
# MAGIC But in comparison to the [`.map()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.map.html) method, [`.apply()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html) is capable of more complicated operations.
# MAGIC
# MAGIC ## Example 3.2 - Applying a function with positional arguments
# MAGIC
# MAGIC If we define a function with an additional positional argument we can still use [`.apply()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html).
# MAGIC See if you can figure out the code below.

# COMMAND ----------

def score_func(x, threshhold):
    if x < threshhold:
        return "Retake"
    else:
        return "Pass"

# COMMAND ----------

# Use .apply() along with score_func 
students_1['result'] = students_1.score.apply(score_func, args=(28,))
print(students_1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 3.3 - Using reducing functions and changing the axis
# MAGIC
# MAGIC In this example we are not applying the function elementwise but rather we are applying a reducing function.
# MAGIC We will also show how we can summarize either by column or by row.  
# MAGIC First let's create a new dataframe containing some fictional test results.

# COMMAND ----------

# run this code

tests = pd.DataFrame(
    {
        "1. Test": [87, 97, 98, 81, 98, 66, 94],
        "2. Test": [66, 76, 73, 79, 55, 98, 84],
        "3. Test": [53, 72, 68, 52, 79, 55, 99],
    }
)
tests.index = ["Robert", "Jana", "Martin", "Kristina", "Peter", "Nikola", "Renato"]
tests

# COMMAND ----------

# MAGIC %md
# MAGIC If we are working with a Pandas DataFrame by default a funciton is applied to each column.
# MAGIC Let's say we want to know the average score for each test.

# COMMAND ----------

# use apply with a reduce function, the result is a series
avg_score = tests.apply(np.mean)
avg_score

# COMMAND ----------

# MAGIC %md
# MAGIC Now we want to find the maximum score each student has achieved. 
# MAGIC In order to achieve this we will use the ``axis`` argument.
# MAGIC
# MAGIC - 0 or 'index': apply function to each column (default)
# MAGIC - 1 or 'columns': apply function to each row.

# COMMAND ----------

tests.apply(np.max, axis = 1)

# COMMAND ----------

# MAGIC %md
# MAGIC The results above could also be achieved using the [``.agg()``](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html) method introduced in an earlier notebook.
# MAGIC However [`.apply()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html) offers more flexibility.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 4.0
# MAGIC
# MAGIC As we already know, regular functions are created using the `def` keyword. These type of functions can have any number of arguments and expressions.

# COMMAND ----------

# Example of regular function
def multi_add(x):
    return x * 2 + 5

# COMMAND ----------

result_1 = multi_add(5)
print(result_1)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Lambda Function
# MAGIC - an anonymous function (it means it can be defined without a name)
# MAGIC - the `def` keyword is not necessary with a lambda function
# MAGIC - lambda functions can have any number of parameters, but the function body can only **contain one expression** (that means multiple statements are not allowed in the body of a lambda function) = it is used for *_one-line expressions_*
# MAGIC - it returns a function object which can be assigned to variable
# MAGIC
# MAGIC General syntax: `lambda x: x`
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Example 4.1

# COMMAND ----------

our_lambda = lambda x: x * 2 + 5
print(our_lambda(5))

# COMMAND ----------

# MAGIC %md
# MAGIC This simple lambda function takes an input `x` (in our case number 5), multiplies it by `2` and adds `5`. <br>
# MAGIC
# MAGIC Lambda functions are commonly used along [`.apply()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html) method and can be really useful. <br>
# MAGIC
# MAGIC ### Example 4.2
# MAGIC
# MAGIC Imagine that the scores of students above have not been correctly recorded and we need to multiply them by 10. 
# MAGIC
# MAGIC Use a lambda function along with [`.apply()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html) and assign it to the specific column of the dataset ('score'). 

# COMMAND ----------

students_1.score = students_1.score.apply(lambda x: x * 10)
print(students_1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3

# COMMAND ----------

# TASK 3 >>>> Use the .apply() method on column 'country' along with lambda to make words uppercase 
#             Do not forget assign it to this column

### Start your code below ###

# COMMAND ----------

# MAGIC %md
# MAGIC We can use lambda functions to simplify Example 3.1 like this:

# COMMAND ----------

# Run this code
students = [(1, 'Robert', 30, 'Slovakia', 26),
           (2, 'Jana', 29, 'Sweden' , 27),
           (3, 'Martin', 31, 'Sweden', 26),
           (4, 'Kristina', 26,'Germany' , 30),
           (5, 'Peter', 33, 'Austria' , 22),
           (6, 'Nikola', 25, 'USA', 23),
           (7, 'Renato', 35, 'Brazil', 26)]

students_1 = pd.DataFrame(students, columns= ['student_id', 'first_name', 'age', 'country', 'score'])

# COMMAND ----------

# A Lambda function is used instead of the custom defined function "score_func"

students_1['result'] = students_1.score.apply(lambda x: "Pass" if (x > 25) else "Retake")
print(students_1)

# COMMAND ----------

# MAGIC %md
# MAGIC # References
# MAGIC
# MAGIC \\(^{1}\\) pandas. pandas.DataFrame.apply. [ONLINE] Available at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html#pandas-dataframe-apply. [Accessed 14 September 2020].
# MAGIC
# MAGIC Stackoverflow. Difference between map, applymap and apply methods in Pandas. [ONLINE] Available at: https://stackoverflow.com/questions/19798153/difference-between-map-applymap-and-apply-methods-in-pandas. [Accessed 14 September 2020].
# MAGIC
# MAGIC Material adapted for RBI internal purposes with full permissions from original authors. Source: https://github.com/zatkopatrik/authentic-data-science
