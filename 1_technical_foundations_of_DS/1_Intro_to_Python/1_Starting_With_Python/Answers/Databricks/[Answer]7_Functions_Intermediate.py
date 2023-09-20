# Databricks notebook source
# MAGIC %md
# MAGIC ## About This Notebook
# MAGIC 
# MAGIC We started to work with functions in previous notebook. We will continue to explore this world by:
# MAGIC - Exploring what happens if we attempt to mess with the ``built-in`` functions.
# MAGIC - Learning what *default arguments* are.
# MAGIC - Finding out how to learn more about particular Python functions through the *documentation*.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Interfering with the Built-in Functions
# MAGIC Let's suppose we create a function called `find_sum()` like the one below:

# COMMAND ----------

def find_sum(my_list):
    a_sum = 0
    for element in my_list:
        a_sum += element
    return a_sum

list_a = [5, 2, 11]
find_sum(list_a)

# COMMAND ----------

# MAGIC %md
# MAGIC The function `find_sum()` does the exact same thing just as the built-in `sum()` function, see below:

# COMMAND ----------

sum(list_a)

# COMMAND ----------

# MAGIC %md
# MAGIC We see that `find_sum()` and `sum()` give out the same result and **we might be tempted to name the find_sum() function simply sum()**.
# MAGIC 
# MAGIC However, using the name ``sum()`` for our function above interferes with the **built-in `sum()` function**. If we name our function `sum()`, and if we try to call `sum()`, Python won't run the built-in `sum()` function anymore, but instead, it will run the `sum()` function that we wrote.
# MAGIC 
# MAGIC We can rewrite our function and return the string "This function does not sum up the number in the list" instead of returning the sum of the elements of the list.

# COMMAND ----------

#the sum() function we wrote takes precedences over the built-in sum() function
def sum(my_list):
    a_string = "This function doesn't not sum up the number in the list"
    return a_string

list_a = [5, 2, 11]
sum(list_a)

# COMMAND ----------

# MAGIC %md
# MAGIC >**We should not use the name of built-in functions to name our functions**. Not only can it confuse your fellow collegues or other people who might read your code later, but also lead to some abnormal function behaviors.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Variable Names and Built-in Functions
# MAGIC 
# MAGIC In the previous session, we learned that we shouln't use the names of built-in functions to name our own functions. Now, let's observe the code below. We first assign `20` (the result of the sum `5 + 15`) to a variable named `sum`. This would then cause interferes with the built-in `sum()` function when we call `sum()`. Python would therefore looks for the value stored in the `sum` variable and instead of calling the built-in function.

# COMMAND ----------

del sum

sum = 5 + 15
sum
list_a = [5, 2, 11]
sum(list_a)

# COMMAND ----------

# MAGIC %md
# MAGIC `sum` is therefore a variable, stored with the integer `20` and when we run `sum(list_a)`, we are running `20(list_1)` instead of the `sum()` function. Hence we got the error message <b>TypeError: 'int' object is not callable</b>.
# MAGIC 
# MAGIC How do we avoid accidentally naming our own functions the same as built-in functions?
# MAGIC A good way to identify a built-in function is by the highlight in the editor.

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC sum() #highlighted 
# MAGIC this_is_a_normal_function() #not highlighted
# MAGIC len() #highlighted 
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC See more built-in functions can be found [here](https://docs.python.org/3/library/functions.html).</p>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Default Arguments
# MAGIC 
# MAGIC When we write a function,  we can initiate parameters with certain default values --- these are known as the default arguments.

# COMMAND ----------

#Initiate the constant parameter with a default argument of 5
def increment_by_5(a, constant = 5):
    return a + constant

increment_by_5(10)

# COMMAND ----------

# MAGIC %md
# MAGIC In the code above, we created the `increment_by_5()` function with two parameters: a and constant. When we call the function, we only need to pass in one positional argument: in the example above, 5.
# MAGIC From the code above we can deduce that the `increment_by_5` function used the argument 5 for the parameter constant.
# MAGIC 
# MAGIC We can also modify our default arguments like below:

# COMMAND ----------

print(increment_by_5(5, constant = 10))
print(increment_by_5(5, constant = 20))
print(increment_by_5(5, constant = 25))

#Alternative:

print(increment_by_5(5, 10))
print(increment_by_5(5, 20))
print(increment_by_5(5, 25))

# COMMAND ----------

# MAGIC %md
# MAGIC If all parameters have default arguments, then it is possible to call a function without passing in any argument. Otherwise we will get an error.

# COMMAND ----------

# all parameters have default arguments
def increment_by_5(a = 1, constant = 5):
    return a + constant

increment_by_5()

# COMMAND ----------

# The first argument does not have a default value
def increment_by_5(a, constant = 5):
    return a + constant

increment_by_5()

# COMMAND ----------

# MAGIC %md
# MAGIC Default arguments can be useful when we use an argument frequently. This can save us some time when we reuse functions. Default arguments are also very useful for building complex functions.
# MAGIC 
# MAGIC We can now try to build a function that opens a CSV file and makes use of default arguments at the same time.
# MAGIC 
# MAGIC ### Task 1.7.3:
# MAGIC 1. Edit the `open_dataset()` function below and add the name of iOS apps data set ('AppleStore.csv') as a default argument for the `file_name` parameter.
# MAGIC 2. Without passing any argument, try to use the `open_dataset()` function to open the AppStore.csv file and assign the data set to a variable named `apps_data`.

# COMMAND ----------

# INITIAL CODE
def open_dataset(file_name='AppleStore.csv'):
    
    opened_file = open(file_name, encoding="utf8")
    from csv import reader
    read_file = reader(opened_file)
    apps_data = list(read_file)
    
    return apps_data

apps_data = open_dataset()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. The Official Python Documentation (IMPORTANT)
# MAGIC 
# MAGIC When you get in touch with more complex Python programming, [the official Python documentation](https://docs.python.org/3/) will become very handy. It is extremely useful for all Python users.
# MAGIC For example if you want to get to know more about the `sum()` function, go to [the search page](https://docs.python.org/3/search.html) and search for "sum" or "sum built-in". Then we can see the search result immediately.
# MAGIC 
# MAGIC The documentation of the ``sum()`` function ends where the next function, ``super()``, begins.
# MAGIC On the first line, we can see all the parameters of the `sum()` function:
# MAGIC `sum(iterable, /, start=0)`
# MAGIC 
# MAGIC We can see how useful the official Python documentation is. We can find information about all the parameters that this function takes as well as its default arguments (try to search for function "open" or "open built-in").
# MAGIC 
# MAGIC > This official Python documentation will be your best friend in your data science journey. It might seem a little bit technical and a bit scary, but don't worry, it is absolutely normal to feel this way. We will definitely have a lot of fun in this Python journey together :)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.7.4:
# MAGIC 1. Find the documentation of the `round()` function. Link to [the search page](https://docs.python.org/3/search.html).
# MAGIC 2. Read and try to understand all the documentation of the `round()` function.
# MAGIC 3. Try to use the right parameters and arguments, can you:
# MAGIC     - Round `3.41` to one decimal point? Assign the result to a variable name of your choice.
# MAGIC     - Round `0.532316` to two decimal points. Assign the result to a variable name of your choice.
# MAGIC     - Round `892.3265621777` to five decimal points. Assign the result to a variable name of your choice.

# COMMAND ----------

# Start your code below:

one_decimal = round(3.41, 1)
two_decimals = round(0.532316, 2)
five_decimals = round(892.3265621777, 5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Multiple Return Statements (OPTIONAL)
# MAGIC 
# MAGIC Is it possible to build a function with multiple return statements?
# MAGIC For example we want to write a function that it's flexible enough to return either the sum of the two numbers or the difference.
# MAGIC 
# MAGIC How can we achieve this? 
# MAGIC 
# MAGIC Fortunately, it is possible to use multiple return statements.
# MAGIC Try to understand the following code:

# COMMAND ----------

def sum_or_difference(x, y, do_sum =True):
    if do_sum: 
        return x + y
    else:
        return x - y
    
print(sum_or_difference(20, 10, do_sum = True))
print(sum_or_difference(20, 10, do_sum = False))

# COMMAND ----------

# MAGIC %md
# MAGIC We have implemented the following logic in the function above:
# MAGIC 
# MAGIC - If do_sum has a value of True, then we return x + y.
# MAGIC - Else (if do_sum is False), we return x - y.
# MAGIC 
# MAGIC We can simply put the function above without using an <b> else </b> statement.

# COMMAND ----------

def sum_or_difference(x, y, do_sum =True):
    if do_sum: 
        return x + y 
    return x - y

print(sum_or_difference(20, 10, do_sum = True))
print(sum_or_difference(20, 10, do_sum = False))

# COMMAND ----------

# MAGIC %md
# MAGIC The above approach works just as the same as the previous function because a function stops executing its rest code as soon as a return statement is executed. Even if there are some other important lines after that `return` statement, it won't get executed.
# MAGIC 
# MAGIC If `do_sum` is `True`, then `return x + y` is executed, so the function stops, and it doesn't execute any of the remaining code.
# MAGIC If `do_sum` is `Flase`, then return `x + y` will not be executed and the function will move forward the eventually reaches the end which is the next return statement: `return a - b`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Returning Multiple Variables
# MAGIC In the previous session, we wrote a function that can either return the sum or the difference of two numbers. 
# MAGIC 
# MAGIC Fortunately, in Python we are able to build functions that can return more than just one variable. See below:

# COMMAND ----------

def sum_and_difference(x, y):
    a_sum = x + y
    difference = x - y
    return a_sum, difference

sum_diff = sum_and_difference(15,5) # Passed 15 and 5 as arguments to the sum_and_difference() function
sum_diff # two values are returned at the end

# COMMAND ----------

# MAGIC %md
# MAGIC We see that both the sum and the difference of our given numbers are returned as the final output. The order of the variables in the return statement is important. For example:

# COMMAND ----------

def sum_and_difference(x, y):
    a_sum = x + y
    difference = x - y
    return difference, a_sum #the order of the return statement is important


sum_diff = sum_and_difference(15,5) 
print(sum_diff)

print(type(sum_diff))

# COMMAND ----------

# MAGIC %md
# MAGIC The order of the final output above is different than the function we had previously. Please pay extra attention to how you construct your return statement.
# MAGIC 
# MAGIC One thing that might surprise you is that the type of the output from the `sum_and_difference` is a <b>tuple </b>, which is a data type that is **very similar to a list**.

# COMMAND ----------

type(sum_diff)

# COMMAND ----------

# MAGIC %md
# MAGIC >A tuple is usually used for storing multiple values. 
# MAGIC 
# MAGIC We can create a tuple just as easy as creating a list, see below:

# COMMAND ----------

this_is_a_list = [1, 'a', 2.5]
this_is_a_tuple = (1, 'a', 2.5)

print(this_is_a_list)
print(type(this_is_a_list))
print()

print(this_is_a_tuple)
print(type(this_is_a_tuple))


# COMMAND ----------

# MAGIC %md
# MAGIC The only difference between constructing a list and a tuple is that a list is surrounded by brackets and a tuple is surrounded by parentheses.
# MAGIC 
# MAGIC >The major difference between tuples and lists is that exisiting values of a list can be modified but those of a tuple cannot.

# COMMAND ----------

this_is_a_list = [1, 'a', 2.5]
this_is_a_tuple = (1, 'a', 2.5)

#Just as lists, tuples support positive and negative indexing.
print(this_is_a_list[0])
print(this_is_a_tuple[0])
print(this_is_a_list[-1])
print(this_is_a_tuple[0])

#A list can be modified
this_is_a_list[0] = 2

# COMMAND ----------

#But on the contrary, a tuple cannot modify the existing values.
this_is_a_tuple[0] = 2

# COMMAND ----------

# MAGIC %md
# MAGIC >The non-modifiable property is called **immutable**. Tuples are one of these immutable data types because we can't change their state after they've been created.
# MAGIC On the other side, lists are mutable data types because we can change their values after they've been created.
# MAGIC 
# MAGIC If we want to modify tuples or any other immutable data types, the only way to change their state is to recreate them.
# MAGIC 
# MAGIC Below is a list of mutable and immutable  data types.
# MAGIC - Mutable: Lists, Dictionaries
# MAGIC - Immutable: Tuples, Integers, Floats, Strings, Booleans.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Functions --- Code Running Quirks
# MAGIC 
# MAGIC Do you know that parameters and return statements are optional for a function? See the example below:

# COMMAND ----------

def print_statement():
    a_string = "This is a function that doesn't have parameters nor a return statement"
    print(a_string)
    
print_statement()

# COMMAND ----------

# MAGIC %md
# MAGIC Functions that don't have a return statement just simply don't return any value.
# MAGIC 
# MAGIC Well, theoretically, they return a <b> `None` </b> value, which just means the absence of a value.

# COMMAND ----------

def print_statement():
    a_string = "This is a function that doesn't have parameters nor a return statement"
    print(a_string)
    
f = print_statement()
print(f)
type(f)

# COMMAND ----------

# MAGIC %md
# MAGIC In the function above, notice that we assigned a text to a variable named `a_string`. What is worth noticing here is that we cannot access `a_string` outside the function. If we try the following, we would get an error:

# COMMAND ----------

def print_statement():
    #Inside the function
    a_string = "This is a function that doesn't have parameters nor a return statement"
    print(a_string)

#Outside the function
print(a_string)

# COMMAND ----------

# MAGIC %md
# MAGIC Note that what's inside a function will only be executed after that function is called. For example, if we have some error inside that function, there won't be any error message raised until we call that function. See example below:

# COMMAND ----------

def do_something():
    "I cannot be divided" / 5
    
print('Code finished running, but no error was raised')

# COMMAND ----------

# MAGIC %md
# MAGIC But we will see the error message if we call that function:

# COMMAND ----------

def do_something():
    "I cannot be divided" / 5
    
print('Now we see the error:')
do_something()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Scopes --- Global and Local
# MAGIC 
# MAGIC Observe the following function carefully:

# COMMAND ----------

def print_constant():
    x = 3.1415926 
    print(x)
    
print_constant()
x

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that we get an error when `x` is called. However, we have already called the function `print_constant()`, **how come `x` is not defined?** `x` is clearly defined inside the `print_constant()` function.
# MAGIC 
# MAGIC The reason for this is that Python saves the variable `x` only temporarily. **Python saves `x` into a kind of temporary memory, which is immediately erased after the `print_constant()` function finishes running**. This explains why we still get the "x is not defined" error when we try to call `x` outside of the `print_constant()` function: it is erased as soon as the `print_constant()` function finishes running.
# MAGIC 
# MAGIC However, this kind of temporary memory storage doesn't apply to the code that is being run outside function definitions.
# MAGIC If we try to define `x = 3.1415926` for example in our main program, or outside function definitions, we can use `x` later on without having any problem or worry about that it was erased from the memory.

# COMMAND ----------

x = 3.1415926

print('x is not erased')
print('we can call x now')

x

# COMMAND ----------

# MAGIC %md
# MAGIC You see that now we don't get an error message by calling `x`.
# MAGIC 
# MAGIC There are several advantages to have such temporary memory associated with a function. For example, if we initialize a variable `x = 123` in the main program and then execute `x = 3.1415926` in the body of a function, the `x` variable of the main program is not being overwritten.

# COMMAND ----------

x = 123

def print_constant():
    x = 3.1415926 
    print(x)
    
print_constant()
x

# COMMAND ----------

# MAGIC %md
# MAGIC You see that with memory isolation, we don't have to worry about overwriting variables from the main program when we write functions, or vice-versa. This can be extremely helpful when we write some large and complicated programs, we don't have to remember all the variable names declared in the main program.
# MAGIC 
# MAGIC >The part of a program where a variable can be accessed is often called <b>scope</b>.
# MAGIC 
# MAGIC ><b>Global scope</b> is known as the variables defined in the main program and <b>local scope</b> is known as the variables defined inside a function.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.7.8
# MAGIC Create a function named `exponential()` that takes in a single parameter named `x`.
# MAGIC - Inside the function definition:
# MAGIC   - Assign `2.71` to a variable named `e`.
# MAGIC   - Print `e`.
# MAGIC - The function should return `e` to the power of `x`.
# MAGIC - Call the `exponential()` function with an argument of `2`. Assign the result to a variable named `result`.
# MAGIC - Hypothesize what you should see if you printed `e` in the main program after calling the `exponential()` function. Print `e` to confirm or reject your hypothesis.

# COMMAND ----------

# Start your code below:

def exponential(x):
    e = 2.71
    print(e)
    return pow(e, x)

result = exponential(2)
print(result)
print(e)  #NameError: name 'e' is not defined

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Scopes --- Searching Order
# MAGIC 
# MAGIC Let's take a look of the code below:

# COMMAND ----------

x = 50
y = 20

def add():
    print(x)
    print(y)
    return x + y

my_result = add()
print(my_result)

# COMMAND ----------

# MAGIC %md
# MAGIC You might be surprised that the code above didn't throw any error.
# MAGIC 
# MAGIC >When a variable is accessed from within a function, Python first searches in the local scope (inside the function's definition) to see if there are any matching variable defined here. If Python doesn't find the variables inside the function's definition, it will continue to search in the global score (the scope of the main program).
# MAGIC 
# MAGIC And now let's take a look of the code below:

# COMMAND ----------

x = 50
y = 20

def add():
    x = 5
    y = 2
    print(x)
    print(y)
    return x+ y

my_result = add()
print(my_result)

# COMMAND ----------

# MAGIC %md
# MAGIC From the code above we see that the local scope is prioritized relative to the global scope. If we define `x` and `y` with different values within the local scope (like `x = 5` and `y = 2`), Python will use those values instead of the `x` and `y` that are lying in the main program, and we'll get a different result.
# MAGIC 
# MAGIC >Pay attention that even though Python searches the global scope if a variable is not found in the local scope, but the reverse doesn't apply here — Python won't search the local scope if a variable is not found in the global scope.
# MAGIC 
# MAGIC Remember the error that we had at the beginning?
# MAGIC To refresh your member, see below:

# COMMAND ----------

def print_constant():
    x = 3.1415926 
    print(x)
    
print_constant()
print(x)

# COMMAND ----------

# MAGIC %md
# MAGIC If this concept is still confusing to you, for a more demonstrative overview, go ahead and check out this [video](https://www.youtube.com/watch?v=r9LtArXOYjk).
