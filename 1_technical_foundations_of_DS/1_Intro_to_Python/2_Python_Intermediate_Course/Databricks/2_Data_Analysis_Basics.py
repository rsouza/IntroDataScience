# Databricks notebook source
# MAGIC %md
# MAGIC # Data Analysis Basics
# MAGIC 
# MAGIC In this notebook, we'll build on the data cleaning we did with the Museum of Modern Art (MoMA) data set in the previous mission, and get into the fun part: analyzing the data! Analyzing the data is extremely important and the techniques that you will learn will be extremely valuable to help you become a data scientist. You can use the techniques to analyze data, to explore data, and to understand data better.
# MAGIC 
# MAGIC There is a cleaned data set called <b>artworks_clean.csv</b> stored in the instance for you. You don't have to re-clean the data again, but we have to convert these values to numeric types so we can do further analysis on them.
# MAGIC ***
# MAGIC ### Task 2.2.0:
# MAGIC Use a `for` loop to iterate over each row in the moma list of lists. Inside the body of the loop:
# MAGIC 
# MAGIC 1. Assign the value from index 6 (Date) to a variable called `date`.
# MAGIC 2. Use an ``if`` statement to check if date is not equal to "".
# MAGIC 3. If date isn't equal to "", convert it to an integer type using the `int()` function.
# MAGIC 4. Finally, assign the value back to index ``6`` in the row.

# COMMAND ----------

from csv import reader

# Read the `artworks_clean.csv` file
opened_file = open('artworks_clean.csv', encoding='utf8')
read_file = reader(opened_file)
moma = list(read_file)
moma = moma[1:]


# Convert the birthdate values
for row in moma:
    birth_date = row[3]
    if birth_date != "":
        birth_date = int(birth_date)
    row[3] = birth_date
    
# Convert the death date values
for row in moma:
    death_date = row[4]
    if death_date != "":
        death_date = int(death_date)
    row[4] = death_date

    
# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Calculating Artist Ages
# MAGIC 
# MAGIC In this session, we want to calculate the ages at which artists created their pieces of art. All we need to do is to subtract the artist's birth year (BeginDate) from the year in which their artwork was created (Date). 
# MAGIC 
# MAGIC Pay attention to the missing values for <b>BeginDate</b>. We'll use a value of 0 for the missing values in BeginDate, but later on we can replace them with something more meaningful.
# MAGIC 
# MAGIC There are a plenty of cases where the artist's age (according to our data set) is very low, including some where the age is negative. We could investigate these specific cases one by one, but since we're looking for a summary, we'll take care of these by categorizing artists younger than 20 as "Unknown" also. This has the handy effect of also categorizing the artists without birth years as "Unknown".
# MAGIC 
# MAGIC To give you a better understanding of some of the values you'll be working with and how they will progress through our code, look at the following table as an example:
# MAGIC 
# MAGIC | Year Artwork Created (date)|Birth Year (birth)|age|final_age|
# MAGIC |- |-|-|-|
# MAGIC |1968|1898 |70|70|
# MAGIC |1931|""|0|"Unknown"|
# MAGIC |1972|1976|-4|"Unknown"|
# MAGIC 
# MAGIC ### Task 2.2.1:
# MAGIC 1. Create an empty list, `ages`, to store the artist age data.
# MAGIC 2. Use a loop to iterate over the `rows` in moma.
# MAGIC 3. In each iteration, assign the artwork year (at index 6) to date and artist birth year (at index 3) to `birth`.
# MAGIC     - If the `birth` date is an int, calculate the age of the artist at the time of creating the artwork, and assign it to the variable `age`.
# MAGIC     - If `birth` isn't an int type, assign 0 to the variable `age`.
# MAGIC     - Append `age` to the `ages` list.
# MAGIC 4. Create an empty list `final_ages`, to store the final age data.
# MAGIC 5. Use a loop to iterate over each `age` in `ages`. In each iteration:
# MAGIC     - If the `age` is greater than 20, assign the `age` to the variable `final_age`.
# MAGIC     - If the `age` is not greater than 20, assign "Unknown" to the variable `final_age`.
# MAGIC     - Append `final_age` to the *final_ages* list.

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Inserting Variables Into Strings (OPTIONAL)
# MAGIC 
# MAGIC Sometimes it is easiser to interpret a value when we insert a list of values into a sentence. For example, if we have some information about one of our collegue's favorite numbers:

# COMMAND ----------

name = "Kellie"
num = 8

# COMMAND ----------

# MAGIC %md
# MAGIC To make the output more understandable, we want to display:

# COMMAND ----------

# MAGIC %md
# MAGIC Kellie's favorite number is 8

# COMMAND ----------

# MAGIC %md
# MAGIC To achieve this, we could use `str()` to convert the integer, and use the `+` operator to combine the values, like this:

# COMMAND ----------

output = name + "'s favorite number is " + str(num)
print(output)

# COMMAND ----------

# MAGIC %md
# MAGIC The[ <b>str.format() method <b>](https://docs.python.org/3/library/stdtypes.html#str.format)is a powerful tool that helps us write easy-to-read code while combining strings with other variables.
# MAGIC 
# MAGIC We use the method with a string — which acts as a template — using the brace characters (`{}`) to signify where we want any variables to be inserted. We then pass those variables as arguments to the method. Let's look at a few examples:

# COMMAND ----------

output = "{}'s favorite number is {}".format("Kellie", 8)
print(output)

# COMMAND ----------

# MAGIC %md
# MAGIC The code is very easy to understand and the `str.format()` method automatically converts the integer to a string for us. The order in which the variables are inserted into the `{}` is by the order of how we pass them as arguments.
# MAGIC 
# MAGIC But if we want to specify the ordering or even to repeat some of the variables, we can do the following:

# COMMAND ----------

output = "{0}'s favorite number is {1}, {1} is {0}'s favorite number".format("Kellie", 8)
print(output)

# COMMAND ----------

# MAGIC %md
# MAGIC However, if we want to make our code even more understandable and clearer, we can give each variable name (using keyword arguments), like this:

# COMMAND ----------

template = "{name}'s favorite number is {num}, {num} is {name}'s favorite number"
output = template.format(name="Kellie", num="8")
print(output)

# COMMAND ----------

# MAGIC %md
# MAGIC The newer versions of Python allow for an even simpler string formatting, using [f-strings](https://docs.python.org/3/tutorial/inputoutput.html#tut-f-strings).  
# MAGIC The next cell shows an example of f-strings in action:

# COMMAND ----------

name = 'Renato'
age = 40

print(f'My name is {name} and i am {age} years old.')

# COMMAND ----------

# MAGIC %md
# MAGIC In the task below, your goal will be to insert an artist's name and birth year into a formatted string. As an example, for the artist René Magritte, the format would be:

# COMMAND ----------

# MAGIC %md
# MAGIC René Magritte's birth year is 1898.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.2.2:
# MAGIC 
# MAGIC We have provided an artist's name and birth year in the `artist` and `birth_year` variables.
# MAGIC 
# MAGIC 1. Create a template string to insert the `artist` and `birth_year` variables into a string, using the format provided above. You may use your choice of the three techniques you learned for specifying which variables goes where.
# MAGIC 2. Use `str.format()` to insert the two variables into your template string, assigning the result to a variable.
# MAGIC 3. Use the `print()` function to call that variable.

# COMMAND ----------

artist = "Pablo Picasso"
birth_year = 1881

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Creating an Artist Summary Function (OPTIONAL)
# MAGIC 
# MAGIC Below we have created a dictionary containing the counts of each artist's artworks. 
# MAGIC Your final task will be to create a function that displays information for a specific artist.
# MAGIC 
# MAGIC This function will take a single argument, which will be the name of an artist, and will display a formatted sentence about that artist.
# MAGIC If we pass "Marc Chagall" into our function as an argument, the final output should be something like: "There are 173 artworks by Marc Chagall in the data set.".
# MAGIC 
# MAGIC Inside the function, we'll need to:
# MAGIC 
# MAGIC - Retrieve the number of artworks by the artist from the `artist_freq` dictionary.
# MAGIC - Define a template for our output.
# MAGIC - Use `str.format() `to insert the artists name and number of artworks into our template.
# MAGIC - Use the `print()` function to display the output.
# MAGIC 
# MAGIC 
# MAGIC ### Task 2.2.3:
# MAGIC 1. Create a function `artist_summary()` which accepts a single argument, the name of an artist.
# MAGIC 2. The function should print a summary of the artist using the steps below:
# MAGIC     - Retrieve the number of artworks from the `artist_freq` dictionary, and assign it to a variable.
# MAGIC     - Create a template string that uses braces (`{}`) to insert the name and variables into the string, using the format from the diagram above.
# MAGIC     - Use `str.format()` method to insert the artist's name and number of artworks into the string template.
# MAGIC     - Use the `print()` function to display the final string.
# MAGIC 3. Use your function to display a summary for the Artist "Henri Matisse".
# MAGIC 
# MAGIC The answer should be: `There are 129 artworks by Henri Matisse in the data set.`

# COMMAND ----------

artist_freq = {}
for element in moma:
    artist = element[1]
    if artist not in artist_freq:
        artist_freq[artist] = 1
    else:
        artist_freq[artist] += 1

# Start your code below: 

