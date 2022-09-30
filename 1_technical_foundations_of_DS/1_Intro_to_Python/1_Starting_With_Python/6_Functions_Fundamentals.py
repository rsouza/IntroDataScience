# Databricks notebook source
# MAGIC %md
# MAGIC ## About This Notebook
# MAGIC 
# MAGIC Functions are the absolute key for any programmer! You might have already noticed that we used for example the `print()` function. This is one of the ``built-in functions`` of python which we can just use right away. We will touch upon these in the beginning of a lecture. Towards the second part of the lecture, we will move towards scenarios when Python does not have the function we need. We will then build our first custom functions.
# MAGIC ***

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Functions
# MAGIC We've learned several useful commands so far: `print()`, `sum()`, `len()`, `min()`, `max()`. These commands are more often known as functions. A function takes in an input, does something to that input, and gives back an output.
# MAGIC 
# MAGIC Take for example the function ``sum()``:
# MAGIC - ``sum()`` takes the input ``list_a``
# MAGIC - it sums up all the values in the list
# MAGIC - it returns the output ``18``, which is the sum of this list

# COMMAND ----------

list_a = [5, 2, 11]
sum(list_a)

# COMMAND ----------

# MAGIC %md
# MAGIC We can understand step 1 and 3 rather quickly, however, what happens inside step 2 is rather ambiguous. 

# COMMAND ----------

# MAGIC %md
# MAGIC First, we have a list about which we want to have some information:
# MAGIC ````python
# MAGIC list_1 = [2, 45,62, -21, 0, 55, 3]
# MAGIC ````
# MAGIC We then initialize a variable named ``my_sum`` with an initial value of 0.
# MAGIC 
# MAGIC ````python
# MAGIC my_sum = 0
# MAGIC ````
# MAGIC 
# MAGIC Afterwards, we loop through the list ``list_1``and sum up all the values in the list one by one:
# MAGIC ````python
# MAGIC for element in list_1:
# MAGIC     my_sum += element
# MAGIC ````
# MAGIC 
# MAGIC At the very end, we print the final result to the screen.
# MAGIC ````python
# MAGIC print(my_sum)
# MAGIC ````
# MAGIC ***
# MAGIC For a better overview:
# MAGIC 
# MAGIC ````python
# MAGIC list_1 = [2, 45, 62, -21, 0, 55, 3]
# MAGIC my_sum = 0
# MAGIC 
# MAGIC for element in list_1:
# MAGIC     my_sum += element
# MAGIC print(my_sum)
# MAGIC 
# MAGIC 
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.6.1:
# MAGIC Let's try to do the same thing with `len()`.
# MAGIC 
# MAGIC 1. Compute the length of the `list_1` without using `len()`.
# MAGIC     - Initialize a variable named length with a value of `0`.
# MAGIC     - Loop through `list_1` and for each iteration add `1` to the current number length.
# MAGIC     
# MAGIC     
# MAGIC 2. Using `len()` function to print the length of ``list_1`` and check whether your function is correct.
# MAGIC > Hint: to better familiarize yourself, you may look up the documentation of ``len()`` function.

# COMMAND ----------

list_1 = [2, 45,62, -21, 0, 55, 3]

#Start your code here:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Built-in Functions
# MAGIC Functions help us tremendously in working faster and simplifying our code. Whenever there is a repetitive task, try to think about using some functions to speed up your workflow.
# MAGIC 
# MAGIC Python has some built-in functions like the ones we have encountered: ``sum()``, ``len()``, ``min()``, and ``max()``. However, Python doesn't have built-in functions for all the tasks we want to achieve. Therefore, it will be necessary to write functions of our own.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Creating Your Own Functions (IMPORTANT)
# MAGIC If we want to create a function named `square()` which performs the mathematical operation of taking a number to the power of 2, how can we achieve this?
# MAGIC To find the square of a number, all we need to do is to multiply that number by itself. For example, to find the square of 4, we need to multiple 4 by itself such as: 4 * 4, which is 16.
# MAGIC So how do we actually create the `square()` function? See below:

# COMMAND ----------

def square(number):
    squared_number = number * number
    return squared_number

# COMMAND ----------

# MAGIC %md
# MAGIC To create the `square()` function above, we:
# MAGIC 
# MAGIC 
# MAGIC 1. Started with the `def` statement
# MAGIC     - Specified the name of the function, which is `square`
# MAGIC     - Specified the name of the variable that will serve as input, which is `number`
# MAGIC     - Surround the input variable `number` within parentheses
# MAGIC     - End the line of the code with a colon `:`
# MAGIC 
# MAGIC 
# MAGIC 2. Specified what we want to do with the input number
# MAGIC     - We first multiplied number by itself: `number * number`
# MAGIC     - We assigned the result of `number * number` to a variable named `squared_number`
# MAGIC 
# MAGIC 
# MAGIC 3. Concluded the function with the `return` statement and specified what to return as the output
# MAGIC     - The output is the variable named `squared_number`, which is the result of `number * number`
# MAGIC 
# MAGIC 
# MAGIC After we have constructed the `square()` function, we can finally put it into practice and compute the square of a number.

# COMMAND ----------

# To compute the square of 2, we use the code
square_2 = square(number = 2)

# To compute the square of 6, we use the code
square_6 = square(number = 6)

# To compute the square of 8, we use the code
square_8 = square(number = 8)

# COMMAND ----------

# MAGIC %md
# MAGIC The variable `number` is the input variable and it can take various values as seen in the code above.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. The Structure of a Function (IMPORTANT)
# MAGIC 
# MAGIC In the previous section we created and used a function named `square()` to compute the square of `2`, `6`, `8`. We have used `number = 2`, `number = 6`, `number = 8` for every input variable:

# COMMAND ----------

def square(number):
    squared_number = number * number
    return squared_number

#To compute the square of 2, we use the code
square_2 = square(number = 2)

#To compute the square of 6, we use the code
square_6 = square(number = 6)

#To compute the square of 8, we use the code
square_8 = square(number = 8)

print(square_2)
print(square_6)
print(square_8)

# COMMAND ----------

# MAGIC %md
# MAGIC To understand what happens when we change the value we assign to `number`, you should try to imagine `number` being replaced with that specific value inside the definition of the function like:

# COMMAND ----------

#For number = 2
def square(number = 2):
    squared_number = number * number
    return squared_number

# COMMAND ----------

#For number = 6
def square(number = 6):
    squared_number = number * number
    return squared_number

# COMMAND ----------

#For number = 8
def square(number = 8):
    squared_number = number * number
    return squared_number

# COMMAND ----------

# MAGIC %md
# MAGIC There are usually three elements that make up the function's definition: The **header** (which contains the def statement), the **body** (where the magic happens), and the **return** statement.

# COMMAND ----------

# MAGIC %md
# MAGIC Note that we indent the body and the return statement four spaces to the right — we did the same for the bodies of for loops and if statements. Technically, we only need to indent at least one space character to the right, but the convention in the Python community is to use four space characters instead. This helps with readability — other people who follow this convention will be able to read your code easier, and you'll be able to read their code easier.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.6.4 (IMPORTANT)
# MAGIC 
# MAGIC Now let's try to recreate what you have learned in the previous chapter, the ``square()`` function.
# MAGIC 
# MAGIC - Create a function `square` with input variable `number` and default value `12`.
# MAGIC - Assign the square of `12` to a variable named `squared_12`.
# MAGIC - Assign the square of `20` to a variable named `squared_20`.
# MAGIC - Print both `squared_12` and `squared_20`.

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Parameters and Arguments (IMPORTANT)
# MAGIC 
# MAGIC We have learned a bit more about functions now. However, instead of `square(number = 6)` we can type just `square(6)` and it will give us the same output.
# MAGIC 
# MAGIC What's behind the scene is that when we use `square(6)` instead of `square(number = 6)`, Python automatically assigns `6` to the `number` variable and it is just the same thing as `square(number = 6)`.
# MAGIC 
# MAGIC Input variables like `number` are more often known as parameters. So `number` is a parameter of the `square()` function and when the parameter `number` takes a value (like `6` in `square(number = 6)`), that value is called an <b>argument</b>.
# MAGIC 
# MAGIC For `square(number=6)`, we'd say the number parameter took in `6` as an argument. For `square(number=1000)`, the parameter number took in `1000` as an argument. For `square(number=19)`, the parameter number took in `19` as an argument, and so on.
# MAGIC 
# MAGIC Now, let's focus on just the return statement. In the `square()` function, we have the return statement as: `return square_number`. However, you can return with an entire expression rather than just a single variable.
# MAGIC 
# MAGIC So instead of saving the result of `number * number` to the variable `squared_number`, we can just return the expression `number * number` and skip the variable assignment step like this:

# COMMAND ----------

#Instead of this:
def square(number):
    squared_number = number * number
    return squared_number

#We can return the expression: number * number
def square(number):
    return  number * number

# COMMAND ----------

# MAGIC %md
# MAGIC The last `square()` function makes our code looks shorter and neater and it is generally encouraged to do so in the practice.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Extract Values From Any Column
# MAGIC Remember our goal of writing our own functions? To have a certain flexibility to speed up our own workflow during the coding process when encountered with some complex problems.
# MAGIC 
# MAGIC Problem 1: Now try to generate a frequency table for a certain column from our data set. 

# COMMAND ----------

# MAGIC %md
# MAGIC In order to simplify and speed up the workflow, we can try to create a separate function for each of these two tasks:
# MAGIC 1. A function that is able to extract the value we desired for any column in a separate list; and
# MAGIC 2. A function that can generate a frequency table for given a list

# COMMAND ----------

# MAGIC %md
# MAGIC In order the solve problem 1, we can first use a function to extract the value for any column we want into a separate list. Then we can pass the output list as an argument to the second function which will give us a frequency table for that given list.
# MAGIC 
# MAGIC How do we extract the values of any column we want from our `apps_data` data set? Please see below:

# COMMAND ----------

#Import data set
opened_file = open('AppleStore.csv', encoding='utf8')
from csv import reader
read_file = reader(opened_file)
apps_data = list(read_file)

#Create an empty list
content_ratings = []

#Loop through the apps_data data set (excluding the header row)
for row in apps_data[1:]:
    #for each iteration, we first store the value from the column we want in a variable
    content_rating = row[10] #(rating has the index number of 10 in each list)
    
    #then we append that value we just stored to the empty list we have 
    #created outside the for loop (content_ratings)
    content_ratings.append(content_rating)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.6.6 (IMPORTANT):
# MAGIC 1. Write a function named ``extract()`` that can extract any column you want from the `apps_data` data set.
# MAGIC     - The function should take in the index `number` of a column as input.
# MAGIC 2. Inside the function's definition:
# MAGIC     - Create an empty list.
# MAGIC     - Loop through the `apps_data` data set and extract only the value you want by using the parameter.
# MAGIC     - Append that value to the empty list.
# MAGIC     - Return the list containing the values of the column.
# MAGIC 3. Use the `extract()` function to extract `cont_rating column` in the data set. Store them in a variable named `app_rating`. The index number of this column is `10`.

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Creating Frequency Tables
# MAGIC 
# MAGIC In this section we will create the second function to our problem 1, which is to create a frequncy table for a given list.

# COMMAND ----------

ratings = ['4+', '4+', '4+', '9+', '9+', '12+', '17+']
#Create an empty list
content_ratings = {}

#Loop through the ratings list 
for c_rating in ratings:
    
    #and check for each iteration whether the iteration variable
    #exists as a key in the dictionary created
    if c_rating in content_ratings: 
        #If it exists, then increment by 1 the dictionary value at that key
        content_ratings[c_rating] +=1
    else:
        #If not, then create a new key-value pair in the dictionary, where the 
        #dictionary key is the iteration variable and the inital dictionary value is 1.
        content_ratings[c_rating] = 1
        
#See the final result after the for loop
content_ratings

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.6.7 (OPTIONAL):
# MAGIC 
# MAGIC 1. Write a function named `freq_table()` which generates a frequency table for any list.
# MAGIC - The function should take in a list as input.
# MAGIC - Inside the function's body, write code that generates a frequency table for that list and stores the table in a dictionary.
# MAGIC - Return the frequency table as a dictionary.
# MAGIC 2. Use the `freq_table()` function on the `app_rating list` (already generated from the previous task, task 5) to generate the frequency table for the `cont_rating` column. Store the table to a variable named `rating_ft`.

# COMMAND ----------

# Complete the code below:

def freq_table(column):
    frequency_table = {}
    for value in column:
        if value in frequency_table:
            frequency_table[value] += 1
        else:
            frequency_table[value] = 1
    return frequency_table

rating_ft = freq_table(app_rating)
print(rating_ft)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Keyword and Positional Arguments
# MAGIC 
# MAGIC There are multiple ways to pass in arguments when a function has more than just one parameters.
# MAGIC 
# MAGIC Take a function named divide(x, y) for example, which takes x, and y as inputs and returns their division.

# COMMAND ----------

def divide(x, y):
    return x / y

# COMMAND ----------

# MAGIC %md
# MAGIC If we want to perform the addition 30 / 5, then we will need to pass 30 and 5 in the parameters of add() function. There are several ways of passing the parameters:

# COMMAND ----------

# Method 1:
divide(x = 30, y = 5)

# COMMAND ----------

# Method 2:
divide(y = 5, x = 30)

# COMMAND ----------

# Method 3:
divide( 30, 5)

# COMMAND ----------

#All of the methods above are correct, however, you cannot do:

divide(5, 30) #it will return you a different result, which is not the same as 30/5

# COMMAND ----------

# MAGIC %md
# MAGIC The syntax `divide(x = 30, y = 5)` and `divide(y = 5, x = 30)` both allow us the pass in the arguments `30` and `5` to correspondingly variable x and variable y. They are also known as named arguments, or more commonly, **keyword arguments**.
# MAGIC 
# MAGIC When we use keyword arguments, the order we use to pass in the arguments doesn't make any difference. However, if we don't specify the keyword argument like in #method 3 and #method 4, then we are not explicit about what arguments correspond to what parameters and therefore we need to pass in the parameters by position. The first argument will be mapped the first parameter, and the second argument will be mapped to the second parameter. These arguments that are passed by position are known as the positional arguments.
# MAGIC 
# MAGIC In the practice, data scientists often use positional arguments because they required less typing and can easily speed up the workflow. So we need to pay extra attention to the order in which we pass on our parameters.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.6.8:
# MAGIC 1. Write an `add()` function that returns the sum of variable x and variable y,
# MAGIC 2. Pass in the parameters by using keyword argument.
# MAGIC 3. Pass in the parameters by using positional argument.
# MAGIC 4. Compare your result of keyword argument and positional argument.

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Combining Functions (OPTIONAL)
# MAGIC 
# MAGIC Do you know that you can use one function inside the body of another function?
# MAGIC 
# MAGIC If we want to write a function called `average()` which takes a list of numbers and returns the average of that list.
# MAGIC To get the mean of a list we first need to sum up all the values in this list and divide by the total number of values in this list.

# COMMAND ----------

def find_sum(my_list):
    a_sum = 0
    for element in my_list:
        a_sum += element
    return a_sum

def find_length(my_list):
    length = 0
    for element in my_list:
        length +=1
    return length

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can use `find_sum()` and `find_length()` inside our `average()` function like this:

# COMMAND ----------

def average(list_of_numbers):
    sum_list = find_sum(list_of_numbers)
    length_of_list = find_length(list_of_numbers)
    mean_list = sum_list / length_of_list
    
    return mean_list

list_a = [5, 2, 11]
average(list_a)

# COMMAND ----------

# MAGIC %md
# MAGIC You can see that we used `find_sum()` and `find_length()` inside the body of the `average()` function. `list_of_numbers` is the parameter which we passed on to the `average()` function, and inside the `average()` function body, it becomes the argument for both `find_sum()` and `find_length()` function. 
# MAGIC 
# MAGIC You may ask why we write `find_sum()` and `find_length()` as two separate functions. The answer is what we learned in the previous session: reusability. Imagine we didn't write those two functions, our `average()` function would look like this:

# COMMAND ----------

def average(list_of_numbers):
    #Finding the sum
    sum = 0
    for element in list_of_numbers:
        sum += element
    
    #Finding the length
    length = 0
    for element in list_of_numbers:
        length +=1
   
    mean_list = sum/length
    
    return mean_list

# COMMAND ----------

# MAGIC %md
# MAGIC Doesn't this function seem a bit long for you? And we would have to write how to find the sum and how to find the length each time we want to perform such action. To write `find_sum()` and `find_length()` outside of the `average()` function enable us to not only use these two functions inside `average()`, but also all the other functions that need these two functions. 
# MAGIC 
# MAGIC Of course we can also shorten our average function like this:

# COMMAND ----------

def average(list_of_numbers):
    return find_sum(list_of_numbers)/find_length(list_of_numbers)

# COMMAND ----------

# MAGIC %md
# MAGIC Which looks super concise and neat. Now let's do a little more practice with writing and combining functions.
# MAGIC 
# MAGIC ### Task 1.6.9:
# MAGIC Write a function named `mean()` that computes the mean for any column we want from a data set. Keep in mind reusability
# MAGIC 
# MAGIC - This function takes in two inputs: a data set and an index value
# MAGIC - Recreate the three functions we've already discussed before:
# MAGIC     - A function `extract()` which takes in a data set and an index and returns the column of the data set corresponding to that index. 
# MAGIC     - A function `find_sum()` which takes in a list and returns the sum of all the elements in the list. 
# MAGIC     - A function `find_length()` which takes in a list and returns the length of that list.
# MAGIC - Inside the body of the `mean()` function, use the `extract()` function to extract the values of a column into a separate list, and compute the mean by using `find_sum()` and `find_length()`.
# MAGIC - The function should then return the mean of the column.
# MAGIC 
# MAGIC Use the `mean()` function to compute the mean of the price column (index number 4) of `apps_data` and assign the result to a variable named `avg_price`.

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Function vs. Method
# MAGIC 
# MAGIC What is actually the difference between function and method? Let's take a closer look.

# COMMAND ----------

# MAGIC %md
# MAGIC Definition of function: 
# MAGIC > A function is a block of code which only runs when it is called. You can pass data, known as parameters, into a function. A function can return data as a result. (w3schools)
# MAGIC 
# MAGIC Remember at the beginning of the chapter we learned that to construct a function we need: **header** (which contains the def statement), **body** (where the magic happens), and **return statement**. You can think of function as it always **returns** something. (In case nothing is explicitly returned, a function automatically returns `None`.)
# MAGIC 
# MAGIC See the example below:

# COMMAND ----------

# test() is a function that prints "test" but returns nothing (or NONE)
def test():
    print("test")

a = test()
print(a)

# COMMAND ----------

# MAGIC %md
# MAGIC There are two types of functions in Python: 
# MAGIC - `User-Defined Functions` (like in the example above)
# MAGIC - `Built-in Functions`
# MAGIC 
# MAGIC `Built-in functions` are the functions that Python provides us with. For example, like the `sum()` function that we encountered previously.
# MAGIC 
# MAGIC See example below:

# COMMAND ----------

my_list = [1,2,3,4,5,6,7,8,9,10]

# sum() is an example of Python built-in function
sum(my_list)

# COMMAND ----------

# MAGIC %md
# MAGIC `Method` is basically a function that is associated with a **class**. Generally, A function is associated with a **super class** - an **object**, as are all things in python.
# MAGIC We will learn more about objects in a later chapter. For now, let's take a peak into what is possible. 
# MAGIC 
# MAGIC The **General Method Syntax** goes like this:

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC class NameOfTheClass:
# MAGIC     def method_name():
# MAGIC     ..................
# MAGIC     # Method_Body
# MAGIC     .............
# MAGIC     
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's see a concrete example of method:

# COMMAND ----------

class ComputerScience(object):
    def my_method(self):
        print("We are learning Python!")
        
        
python = ComputerScience()
python.my_method()

# COMMAND ----------

# MAGIC %md
# MAGIC In the example above we have defined a class called `ComputerScience`. Afterwards we created an object called `python` from the class blueprint. We then called our custom-defined method called `my_method` with our object `python`.
# MAGIC 
# MAGIC Do you now see the difference?
# MAGIC 
# MAGIC >Differently than a function, methods are called on an object. 
# MAGIC 
# MAGIC Like in the example above, we called our method `my_method` from our object `python`.
