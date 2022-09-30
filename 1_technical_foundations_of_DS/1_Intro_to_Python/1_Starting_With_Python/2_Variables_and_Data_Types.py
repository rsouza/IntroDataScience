# Databricks notebook source
# MAGIC %md
# MAGIC ## About This Notebook
# MAGIC 
# MAGIC Probably the most useful concept within programming is that of **variables**. We will therefore focus on variables within this notebook. We will begin by exploring why we need them and how are they created. Later on, we will move onto **data types**. These mean that variables can be of different type depending on what they store.
# MAGIC 
# MAGIC Alongside, we will learn some useful tricks, such as *conversion* betweeen data types.
# MAGIC ***

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Saving Values
# MAGIC We have learned the basics of Python programming and performed a couple of arithmetical operations in Python. However, how do we actually save values and work with numerical and text data? For example, if we want to save the result of an arithmetical operation for a different execution. Let's say `(20-10)*2 = 20`, and we want to save 20 as our result. We can therefore write:

# COMMAND ----------

result = 20

# COMMAND ----------

# MAGIC %md
# MAGIC If we print the result, the output is: 20

# COMMAND ----------

result = 20
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC We can also directly save (20-10)*2 instead of saving 20:

# COMMAND ----------

result = (20-10)*2
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC Pay attention that, when we print(result), the output is the value of calculation and not `(20-10) * 2`. **The computer first calculates (20-10) * 2 and then saves the result 20 to variable name "result".**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.2.1:
# MAGIC 1. Save the result of (50 + 6)*32 to the variable name `result`.
# MAGIC 2. Print result.

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Variables
# MAGIC When we run the code result = 20, the value 20 is saved in the computer memory. The computer memory has many storage locations, and the value 20 is saved to one particular location.
# MAGIC 
# MAGIC The value we just saved, which is 20, has a unique identifier in the storage location which we can use it to access 20. We can use the identifier *result* to access 20 in our program. For example:

# COMMAND ----------

result = 20
print(result)
print(result * 2)
print(result + 1)

# COMMAND ----------

# MAGIC %md
# MAGIC This unique identifier is commonly known as a **variable**. 
# MAGIC *result = 20* ran after we hit the Run button, the computer stored 20 in a variable or a storage location named *result* based on our command --- therefore "result" is a **variable name**.
# MAGIC Notice that the order of the variable naming is very important. The variable name is to the left of the = sign and the value we want to store to this variable name is located to the right.
# MAGIC Therefore if we want to store the value 20 to a variable named result, result = 20 must be written and not 20 = result.

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's do some practice declaring some variable names.
# MAGIC 
# MAGIC ### Task 1.2.2:
# MAGIC 1. Store the value 10 in a variable named some_value.
# MAGIC 2. Store the result of (38+6-2)*23 to a variable named some_result.
# MAGIC 3. Use the `print()` command to display the following:
# MAGIC   * The value stored in the some_value variable.
# MAGIC   * The result of adding 8 to the variable some_result.
# MAGIC   * The result of adding some_value to some_result.

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Variable Names
# MAGIC In the previous task, we have declared some variable names. We also learned that we can choose names for our variables. However, the names we choose must comply with a number of syntax rules. For example, if we try to name a variable `a result`, a syntax error will occur because we're not allowed to use space in variable names.

# COMMAND ----------

a result = 20

# COMMAND ----------

# MAGIC %md
# MAGIC There are two syntax rules we must follow when we declare our variables: <br>
# MAGIC 1) Only letters, numbers, or underscores (we can't use apostrophes, hyphens, whitespace characters, etc.) can be used. <br>
# MAGIC 2) Variable names cannot start with a number.<br>
# MAGIC 
# MAGIC For example, errors will occur if we pick any of the following variable names:

# COMMAND ----------

1_result = 2
new result = 3
sister's_vacation_day = 23
old-result = 20
price_in_$ = 20

# COMMAND ----------

# MAGIC %md
# MAGIC All variable names are case sensitive, notice that result is different from a variable named Result:

# COMMAND ----------

result = 20
Result = 30
print(result)
print(Result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Updating Variables
# MAGIC 
# MAGIC The value saved in a variable can be changed or updated.
# MAGIC For example, in the code below we have first stored 2 in the variable result and then we update result to store 20 instead.

# COMMAND ----------

result = 2
print(result)
print(result + 1)

result = 20
print(result)

result = result + 10
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC Pay attention to the code above:
# MAGIC - The variable result initially only stores a value of 2
# MAGIC - `result + 1` equals to 3 because `result` restores a value of 2 --- so `result + 1` becomes 2 + 1
# MAGIC - when we run `result = result + 10`, `result` is updated to store of value of `result + 10`, which is 30. It is the same as running `result = 20 + 10` because `result` has a value of 20.
# MAGIC `print(result)` outputs 30 after we executed result = result + 10.
# MAGIC Now let's have a little practice with variables.
# MAGIC 
# MAGIC 
# MAGIC ### Task 1.2.4:
# MAGIC 1.  Update the variable income by adding 5000 to its current value.
# MAGIC 2.  Print income

# COMMAND ----------

income = 2000

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC **Pay attention to how we updated the variable, for example, by using x = x + 1. It is different than what we normally follows in mathematics. x = x +1 would be a false statement because x can never be equal to x + 1.** This means that the = operator sign in python or in any programming language in general doesn't have the same meaning as it does in mathematics.
# MAGIC 
# MAGIC **In Python, the = operator means assignment**: the value on the right is assigned to the variable on the left, just like how we name our variable. It doesn't mean equality. We call = an assignment operator, and we read code like x = 2 as "two is assigned to x" or "x is assigned two," but not "x equals two."

# COMMAND ----------

# MAGIC %md
# MAGIC By the way, Python offers a shortcut for inplace operations +=, -=, /= and *=

# COMMAND ----------

x = 10
print(x)

x += 1    # is equivalent to x = x + 1
print(x)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Types of Values
# MAGIC We have only worked with integers so far, of course we can also work with decimal numbers in Python. To understand what type of numbers or even values you are working with in Python, we can simply use the `type()` command to see the type of a value. For example:

# COMMAND ----------

print(type(10))
print(type(10.))
print(type(10.0))
print(type(2.5))

# COMMAND ----------

# MAGIC %md
# MAGIC We see that Python distinguishes integers and decimal numbers since the integer 10 has the <b> int </b> type and the decimal numbers 10., 10.0, and 2.5 have the <b> float</b> type. All integers have the <b> int </b> type, and numbers that has a decimal point have the <b> float </b>type. 
# MAGIC 
# MAGIC Even though these numbers are classified into different <b> types</b> or have different <b> data types </b>, we can still perform arithmetical operations with them. For example, we can still add an <b> int </b> data type to a <b> float </b> data type.

# COMMAND ----------

print(10 + 10.0)
print(2.5 * 5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.2.5:
# MAGIC 1. Assign a value of 10 to a variable named `value_1` <br>
# MAGIC 2. Assign a value of 20.5 to a variable named `value_2` <br>
# MAGIC 3. Update the value of `value_1` by adding 2.5 to its current value. Try to use the syntax shortcut like += operator. <br>
# MAGIC 4. Update the value of `value_2` by multiplying its current value by 5. Try to use the syntax shorcut like *= operator. <br>
# MAGIC 5. Print the result of `value_1` and `value_2` by using `print()` command. <br>

# COMMAND ----------

#Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Conversion
# MAGIC 
# MAGIC Is it possible to convert one type of value to another type of value? Totally! For example, <b> float()</b> command is used to convert an integer to a float and  <b>int()  </b>command is to convert a float to an integer.

# COMMAND ----------

float(10)

# COMMAND ----------

int(2.6)

# COMMAND ----------

# MAGIC %md
# MAGIC Pay attention to the `int()` command and notice that 2.6 is rounded down to a 2. The `int()` command will always round a float value down, even if the number after the decimal point is greater than five.
# MAGIC 
# MAGIC However, we can also use `round()` command to properly round off a number, which follows the normal rounding rule.

# COMMAND ----------

round(2.3)

# COMMAND ----------

round(2.5)

# COMMAND ----------

round(2.99)

# COMMAND ----------

# MAGIC %md
# MAGIC It is possible and often encouraged to combine commands. For example, we can use `round()` inside a `print()` command. Notice the different output printed on the screen between a simple `round()` command and `print(round())`.

# COMMAND ----------

print(round(2.3))
print(round(2.5))
print(round(2.99))

# COMMAND ----------

round(2.3)
round(2.5)
round(2.99)

# COMMAND ----------

# MAGIC %md
# MAGIC Another detail to pay attention to is that `round()` command doesn't change the value stored by a variable.

# COMMAND ----------

variable_1 = 2.5
print(round(variable_1))
print(variable_1)

# COMMAND ----------

# MAGIC %md
# MAGIC However, if we assign the rounded value back to the variable, we are able to change the value stored in the variable.

# COMMAND ----------

variable_1 = round(2.5)
print(variable_1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.2.6:
# MAGIC 1. Assign a value of 11.2 to a variable named `value_1` <br>
# MAGIC 2. Assign a value of 2.5 to a variable named `value_2` <br>
# MAGIC 3. Round the value of `value_1` by using the `round()` command and assign the rounded value back to `value_1` <br>
# MAGIC 4. Convert the value of `value_2` from a float to an integer value using the `int()` command and assign the value back to `value_2` <br>
# MAGIC 5. Print the result of `value_1` and `value_2` by using the `print()` command. <br>

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Strings
# MAGIC 
# MAGIC Up till now, we have only worked with <b> int </b> and <b> float</b> values. But in computer programming there are many more types of values. 
# MAGIC Take a look at the table down below:
# MAGIC 
# MAGIC 
# MAGIC | Track_name |  Price |  Currency |  Rating_count_total | User_rating|
# MAGIC |------------|:------:|----------:|---------------------:|-----------:|
# MAGIC | Facebook | 0.0 | USD | 2974676 | 3.5|
# MAGIC | Instagram |    0.0  |   USD |2161558 |4.5|
# MAGIC | Clash of Clans | 0.0|    USD | 2130805 |4.5|
# MAGIC | Temple Run |    0.0  |   USD |1724546 |4.5|
# MAGIC | Pandora - Music & Radio | 0.0|    USD | 1126879 |4.5|
# MAGIC 
# MAGIC Data Source:  [Mobile App Store Data Set (Ramanathan Perumal)](https://www.kaggle.com/ramamet4/app-store-apple-data-set-10k-apps)</p>

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see that the columns <b> track_name</b> and <b> currency </b> consist of text and not numbers. In Python, we can create text by using the quotation marks (" "):

# COMMAND ----------

app_name = "Instagram"
currency = "USD"

print(app_name)
print(currency)

# COMMAND ----------

# MAGIC %md
# MAGIC Both double quotation marks (" ") and single quotation mark (' ') are allowed in the Python syntax.
# MAGIC To create the word "Instagram", we can use either "Instagram", or 'Instagram'. The values surrounded by quotation marks are called strings and are represented in Python by the <b>str</b> type.

# COMMAND ----------

type('Instagram')

# COMMAND ----------

# MAGIC %md
# MAGIC However, strings are not only limited to letters. It is also possible to use numbers, space, or other characters. See example below:

# COMMAND ----------

bank = 'My Bank'
number = 'number is 123456789'

print(bank)
print(number)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.2.7:
# MAGIC 1. Assign the string `"Instagram"` to a variable named `app_name`. <br>
# MAGIC 2. Assign  the string  `"4.5"` to a variable named `average_rating`. <br>
# MAGIC 3. Assign the string `"2161158"` to a variable named `total_ratings`. <br>
# MAGIC 4. Assign the string `"data"` to a variable named `value`. <br>
# MAGIC 5. Display `app_name` variable using `print()` command. <br>

# COMMAND ----------

# Start your code below:

