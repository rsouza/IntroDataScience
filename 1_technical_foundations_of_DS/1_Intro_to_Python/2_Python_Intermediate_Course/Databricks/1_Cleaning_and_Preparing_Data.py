# Databricks notebook source
# MAGIC %md
# MAGIC # Cleaning and Preparing Data in Python
# MAGIC 
# MAGIC A lot of what Data Scientists do is about cleaning data. In this following notebook, you will be going over some basic steps on hwo to do this.
# MAGIC ***

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Reading our MoMA Data Set
# MAGIC 
# MAGIC This time we will work with data from the Museum of Modern Art (MoMA), a museum with one of the largest collections of modern art in the world in the center of New York City. Each row in this table represents a unique piece of art from the Museum of Modern Art. Let's take a look at the first five rows:
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC | Title |  Artist |  Nationality | BeginDate|EndDate|Gender|Date|Department|
# MAGIC |------------|:------:|----------:|---------------------:|-----------:|----------:|---------------------:|-----------:|
# MAGIC | Dress MacLeod from Tartan Sets |Sarah Charlesworth |(American) | (1947) | (2013)|(Female)|1986|Prints & Illustrated Books|
# MAGIC |Duplicate of plate from folio 11 verso (supplementary suite, plate 4) from ARDICIA|   Pablo Palazuelo |  (Spanish)|(1916) |(2007)|(Male)|1978|Prints & Illustrated Books|
# MAGIC |Tailpiece (page 55) from SAGESSE | Maurice Denis| (French) | (1870)|(1943)|(Male)|1889-1911|Prints & Illustrated Books|
# MAGIC |Headpiece (page 129) from LIVRET DE FOLASTRIES, À JANOT PARISIEN |Aristide Maillol| (French) |	(1861) |(1944)|(Male)|1927-1940|Prints & Illustrated Books|
# MAGIC |97 rue du Bac| Eugène Atget| (French) |(1857) |(1927)|(Male)|1903|	Photography|
# MAGIC 
# MAGIC The MoMA data is in a ``CSV`` file called ``artworks.csv``. Below is a short explanation of some of the variable names that you encountered above.
# MAGIC 
# MAGIC - `Title:` The title of the artwork.
# MAGIC - `Artist:` The name of the artist who created the artwork.
# MAGIC - `Nationality:` The nationality of the artist.
# MAGIC - `BeginDate:` The year in which the artist was born.
# MAGIC - `EndDate:` The year in which the artist died.
# MAGIC - `Gender:` The gender of the artist.
# MAGIC - `Date:` The date that the artwork was created.
# MAGIC - `Department:` The department inside MoMA to which the artwork belongs.
# MAGIC 
# MAGIC How do we access the csv file using Python then?
# MAGIC Just like we learned in the first course, **Python has a built-in csv module that can handle the work of opening a CSV for us**.

# COMMAND ----------

#First, import the reader() function from the csv module:
from csv import reader

#Second, use the Python built-in function open() to open the Artworks.csv file:
opened_file = open('Artworks.csv')

#Then use reader() to parse (or interpret) the opened_file:
read_file = reader(opened_file)

#Use the list() function to convert the read_file into a list of lists format:
artworks = list(read_file)

#Finally, remove the first row of the data, which contains the column names:
artworks = artworks[1:]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Replacing Substrings with the replace Method
# MAGIC 
# MAGIC Sometimes when we're cleaning data, some parts of strings need to be replace in order to make our data look clean and consistent.
# MAGIC 
# MAGIC The technique we will learn in this section is called <b>substring</b>. 
# MAGIC >For example, if we have a string "Swimming is my favorite activity" and we want to change "Swimming" to "Running", with the substring technique, the sentence will look like this: "Running is my favorite activity".
# MAGIC 
# MAGIC In order to do this, we'll need to use the `str.replace()` function. The following steps take place:
# MAGIC 1.  to find all instances of the old substring, which in our example "Swimming".
# MAGIC 2.  to replace each of those instances with the new substring, "Running".
# MAGIC 
# MAGIC `str.replace()` takes two arguments:
# MAGIC 1. old: The substring we want to find and replace.
# MAGIC 2. new: The substring we want to replace old with.
# MAGIC 
# MAGIC When we use `str.replace()`, we substitute the str for the variable name of the string we want to modify. Let's look at an example in code:

# COMMAND ----------

fav_activity = "Swimming is my favorite activity."
print(fav_activity) 
fav_activity = fav_activity.replace("Swimming", "Running")
print(fav_activity)

# COMMAND ----------

# MAGIC %md
# MAGIC In the code above, we:
# MAGIC 
# MAGIC - Created the original string and assigned it to the variable name ``fav_activity``.
# MAGIC - Replaced the substring "Swimming" with the substring "Running" by calling `fav_activity.replace()`.
# MAGIC - Assigned the result back to the original variable name using the `=` sign.
# MAGIC 
# MAGIC There is something to pay attention to is that when we use `str.replace()`, this function will replace all instances of the substring it finds. See the following example:

# COMMAND ----------

fav_activity = fav_activity.replace("i", "I")
print(fav_activity)

# COMMAND ----------

# MAGIC %md
# MAGIC You see that in the code above, all "i"s in the fav_activity are replaced with "I". This is something to keep in mind when we use `str.replace()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.1.2:
# MAGIC 
# MAGIC In the text editor below, we have created a string variable `string1` containing the string `"I am awesome"`.
# MAGIC Now use the `str.replace()` method to create a new string, `string2`:
# MAGIC - The new string should have the value `"I am amazing"`.

# COMMAND ----------

string1 = "I am awesome."

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Cleaning the Nationality and Gender Columns
# MAGIC 
# MAGIC Now let's see how we can use the `str.replace()` method on a bigger data set. I have a shortened version of our data set below:
# MAGIC 
# MAGIC | Title |  Artist |  Nationality |Gender|
# MAGIC |------------|:------:|----------:|---------------------:
# MAGIC | Dress MacLeod from Tartan Sets |Sarah Charlesworth |(American) | (1947) | (2013)|(Female)|
# MAGIC |Duplicate of plate from folio 11 verso (supplementary suite, plate 4) from ARDICIA|   Pablo Palazuelo |  (Spanish)|(Male)|
# MAGIC |Tailpiece (page 55) from SAGESSE | Maurice Denis| (French) |(Male)|
# MAGIC |Headpiece (page 129) from LIVRET DE FOLASTRIES, À JANOT PARISIEN |Aristide Maillol| (French) |(Male)|
# MAGIC |97 rue du Bac| Eugène Atget| (French) |(Male)|
# MAGIC 
# MAGIC Do you see that the Nationality and Gender columns have parentheses (()) at the start and the end of the values? In this session, we want to learn how to remove those values.
# MAGIC 
# MAGIC In the session, we learned how to use `str.replace()` to replace one substring with another. What we want, however, is to remove a substring, not replace it. **In order to remove a substring, all we need to do is to replace the substring with an empty string: `""`**.
# MAGIC 
# MAGIC We need to perform this action many times in order to replace all unwanted characters in our whole moma data set. We can do this with a for loop. Let's see an example using a small sample of our data:

# COMMAND ----------

nationalities = ['(American)', '(Spanish)', '(French)']

for n in nationalities:
    clean_open = n.replace("(","")
    print(clean_open)

# COMMAND ----------

# MAGIC %md
# MAGIC We removed the `(` character from the start of each string. In order to remove both, we'll have to perform the `str.replace()` twice:

# COMMAND ----------

nationalities = ['(American)', '(Spanish)', '(French)']

for n in nationalities:
    clean_open = n.replace("(","")
    clean_both = clean_open.replace(")","")
    print(clean_both)

# COMMAND ----------

# MAGIC %md
# MAGIC How can we adopt this code to work on the whole data set? We'll start by printing the value from the Nationalities column (with a column index `4`) for three rows in our moma data set. We'll use the same rows after our loop so we can see how the values changed:

# COMMAND ----------

# Read in csv file
from csv import reader
opened_file = open('Artworks.csv',encoding="utf-8")
read_file = reader(opened_file)
moma = list(read_file)
moma = moma[1:]

print(moma[200][4])
print(moma[400][4])
print(moma[800][4])

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we'll loop over each row in the moma data set. In each row, we'll:
# MAGIC 
# MAGIC - Assign the Nationality value from index `4` to a variable name.
# MAGIC - Use `nationality.replace()` to remove all instances of the open parentheses.
# MAGIC - Use `nationality.replace()` to remove all instances of the close parentheses.
# MAGIC - Assign the cleaned nationality back to row index `4`.

# COMMAND ----------

for row in moma:
    nationality = row[4]
    nationality = nationality.replace("(","")
    nationality = nationality.replace(")","")
    row[4] = nationality

# COMMAND ----------

# Let's look at the values of the same three rows after our code:

print(moma[200][4])
print(moma[400][4])
print(moma[800][4])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.1.3:
# MAGIC Now it's your turn — you'll be removing the parentheses from both the `Nationality` and `Gender` columns.
# MAGIC Gender information you can find at index `7` of the row.

# COMMAND ----------

# Variables you create in previous screens
# are available to you, so you don't need
# to read the CSV again.

# Start your code here:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. String Capitalization
# MAGIC 
# MAGIC The Gender column in our data set contains four unique values:
# MAGIC 
# MAGIC - (an empty string)
# MAGIC - "Male"
# MAGIC - "Female"
# MAGIC - "male"

# COMMAND ----------

# MAGIC %md
# MAGIC In our data set, there are two different capitalizations used in our data set for "male." This could be caused by manual data entry. Different people could use different capitalizations when they enter words.
# MAGIC 
# MAGIC There are a few ways we could handle this using what we know so far:
# MAGIC 
# MAGIC 1. We could use ``str.replace()`` to replace m with ``M``, but then we'd end up with instances of FeMale.
# MAGIC 2. We could use ``str.replace()`` to replace male with ``Male``. This would also give us instances of FeMale.
# MAGIC 
# MAGIC However, here comes the problem: even if the word "male" wasn't contained in the word "female," both of these techniques wouldn't be good options if we had a column with many different values, like our Nationality column. Instead, what we should use is the <b>str.title()</b> method.
# MAGIC > ``str.title()``: a Python string method designed specifically for handling capitalization. The method returns a copy of the string with the first letter of each word transformed to uppercase (also known as <b>title case</b>).
# MAGIC 
# MAGIC Let's take a look at an example:

# COMMAND ----------

my_string = "The cool thing about this string is that it has a CoMbInAtIoN of UPPERCASE and lowercase letters!"

my_string_title = my_string.title()
print(my_string_title)

# COMMAND ----------

# MAGIC %md
# MAGIC Using title case will give us consistent capitalization for all values in the Gender column.
# MAGIC 
# MAGIC We have a number of rows containing an empty string (`""`) for the Gender column. This could be a result of:
# MAGIC 
# MAGIC - The person entering the data has no information about the gender of the artist.
# MAGIC - The artist is unknown and so is the gender.
# MAGIC - The artist's gender is non-binary.
# MAGIC 
# MAGIC Now let's try to use this technique to make the capitalization of both the Nationality and Gender columns uniform. The Nationality column has 84 unique values, so to help you, we'll provide you with the values that aren't already in title case:
# MAGIC 
# MAGIC - `''`
# MAGIC - `'Nationality unknown'`
# MAGIC 
# MAGIC ### Task 2.1.4:
# MAGIC 
# MAGIC Use a loop to iterate over all rows in the moma list of lists. For each row:
# MAGIC 
# MAGIC 1. Clean the Gender column.
# MAGIC     - Assign the value from the Gender column, at index ``7``, to a variable.
# MAGIC     - Make the changes to the value of that variable.
# MAGIC         - Use the `str.title()` method to make the capitalization uniform.
# MAGIC         - Use an if statement to check if the value is an empty string. If the value is an empty string, give it the value `"Gender Unknown/Other"`.
# MAGIC     - Assign the modified variable back to list index `7` of the row.
# MAGIC 2. Clean the Nationality column of the data set (found at index `4`) by repeating the same technique you used for the Gender column.
# MAGIC     - For missing values in the `Nationality` column, use the string `"Nationality Unknown"`.

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Errors During Data Cleaning
# MAGIC 
# MAGIC We have analyzed the artist nationalities. Now let's have a look at the <b>BeginDate</b> and <b>EndDate</b> columns
# MAGIC 
# MAGIC These two columns contain the birth and death dates of the artist who created the work. Let's take a look at the column:

# COMMAND ----------

for row in moma[:5]:
    birth_date = row[5]
    death_date = row[6]
    print([birth_date, death_date])

# COMMAND ----------

# MAGIC %md
# MAGIC These values are wrapped in parentheses as four-digit strings. How can we clean these columns? We need to:
# MAGIC - Remove the parentheses from the start and the end of each value.
# MAGIC - Convert the values from the string type to an integer type. This will help us perform calculations with them later.
# MAGIC 
# MAGIC In the previous two screens, we had to repeat code twice — first when we cleaned the Gender column, and then when we cleaned the Nationality column. If we don't want to keep repeating code, we can create a function that performs these operations, then use that function to clean each column.

# COMMAND ----------

def clean_and_convert(date): #Takes a single argument
    date = date.replace("(", "") #Uses str.replace() to remove the "(" character

    date = date.replace(")", "") #Uses str.replace() to remove the ")" character
    date = int(date) #Convert the string to an integer
    return date

# COMMAND ----------

# MAGIC %md
# MAGIC If we have a ``BeginDate`` value of '(1958)':

# COMMAND ----------

birth_date = '(1958)'
cleaned_date = clean_and_convert(birth_date)
print(cleaned_date)
print(type(cleaned_date))

# COMMAND ----------

# MAGIC %md
# MAGIC Our function successfully removes the parentheses and converts the value to an integer type. Unfortunately, our function won't work for every value in our data set. If we have two values at the same time:

# COMMAND ----------

row_43 = moma[42] # row 43
print(row_43)

# COMMAND ----------

#This will throw an error

birth_date = '(1936) (0) (1936) (1931) (1931) (1944)'
cleaned_date = clean_and_convert(birth_date)

# COMMAND ----------

# MAGIC %md
# MAGIC Our code has not completed successfully, instead returning a `ValueError`. As we learned in the previous course, the name for the error message is called a traceback. The final line of the traceback tells us the underlying error:

# COMMAND ----------

# MAGIC %md
# MAGIC ValueError: invalid literal for int() with base 10: '1936 0 1936 1931 1931 1944'

# COMMAND ----------

# MAGIC %md
# MAGIC One way to handle these scenarios is to use an if statement to make sure we aren't encountering an empty string before we convert our value.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Parsing Numbers from Complex Strings, Part One
# MAGIC 
# MAGIC We have successfully converted the ``BeginDate`` and ``EndDate`` columns into numeric values. If we were to combine the data from the BeginDate column (the artist's year of birth) with the data in the Date column (the year of creation) we can therefore calculate the age at which the artist produced this piece of artwork.
# MAGIC 
# MAGIC That means we need to clean the data in the `Date` column in order to perform such calculation as mentioned above.
# MAGIC 
# MAGIC Let's examine a sample of the types of values contained in this column:

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC 1912
# MAGIC 1929
# MAGIC 1913-1923
# MAGIC (1951)
# MAGIC 1994
# MAGIC 1934
# MAGIC c. 1915
# MAGIC 1995
# MAGIC c. 1912
# MAGIC (1988)
# MAGIC 2002
# MAGIC 1957-1959
# MAGIC c. 1955.
# MAGIC c. 1970's
# MAGIC C. 1990-1999
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC This column contains data in many different formats:
# MAGIC 
# MAGIC - Some years are in parentheses.
# MAGIC - Some years have c. or C. before them, indicating that the year is approximate.
# MAGIC - Some have year ranges, indicated with a dash.
# MAGIC - Some have 's to indicate a decade.
# MAGIC 
# MAGIC In this session,we want to to remove all the extra characters and be left with either a range or a single year. We will then finish processing the data in the sessions that follow. For the two special cases listed above:
# MAGIC 
# MAGIC - Where there is an 's that signifies a decade, we'll use the year without the 's.
# MAGIC - Where c. or similar indicates an approximate year, we'll remove the c. but keep the year.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.1.6 (OPTIONAL):
# MAGIC 1. Create a function called `strip_characters()`, which accepts a string argument and:
# MAGIC     - Iterates over the `bad_chars` list, using `str.replace()` to remove each character.
# MAGIC     - Returns the cleaned string.
# MAGIC 2. Create an empty list, `stripped_test_data`.
# MAGIC 3. Iterate over the strings in `test_data`, and on each iteration:
# MAGIC     - Use the function you created earlier to clean the string.
# MAGIC     - Append the cleaned string to the `stripped_test_data` list.

# COMMAND ----------

test_data = ["1912", "1929", "1913-1923",
             "(1951)", "1994", "1934",
             "c. 1915", "1995", "c. 1912",
             "(1988)", "2002", "1957-1959",
             "c. 1955.", "c. 1970's", 
             "C. 1990-1999"]


bad_chars = ["(",")","c","C",".","s","'", " "]

# Start your code here:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Parsing Numbers from Complex Strings, Part Two

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's take a look at the result from your previous task:

# COMMAND ----------

# MAGIC %md
# MAGIC 1912
# MAGIC 1929
# MAGIC 1913-1923
# MAGIC 1951
# MAGIC 1994
# MAGIC 1934
# MAGIC 1915
# MAGIC 1995
# MAGIC 1912
# MAGIC 1988
# MAGIC 2002
# MAGIC 1957-1959
# MAGIC 1955
# MAGIC 1970
# MAGIC 1990-1999

# COMMAND ----------

# MAGIC %md
# MAGIC There are two different scenarios that we need to pay attention to when we are converting them into integers:
# MAGIC - we have values in a single year, like 1912
# MAGIC - we also have values in ranges of years, like 1913-1923

# COMMAND ----------

# MAGIC %md
# MAGIC As a data scientist, you need to make decisions on how you will structure your code. One option could be to discard all approximate years so we know that our calculations are exact. For example, when we're calculating an artist's age, an approximate age is also acceptable (the difference between 30 and 33 years old is more nuanced than we need).
# MAGIC 
# MAGIC Whichever way you decide to proceed isn't as important as thinking about your analysis and having a valid reason for this particular decision.
# MAGIC 
# MAGIC So this is what we will do:
# MAGIC - when we have values in a single year, like 1912, we'll keep it as it is.
# MAGIC - when we also have values in ranges of years, like 1913-1923, we'll average the two years.
# MAGIC 
# MAGIC How do we proceed with our above decision? We can do the following:
# MAGIC 1. Have an if statement to check if there is a dash character ``-`` in the string, so we know if it's a range or not.
# MAGIC 2. If the date is a range:
# MAGIC     - Split the string into two strings, take the first part (before the dash), and the second part (after the dash)
# MAGIC     - Convert the two numbers into integer type
# MAGIC     - Take the average of those two numbers
# MAGIC     - Use the round() function to round the average
# MAGIC 3.  If the date isn't a range:
# MAGIC     - Convert the value to an integer type.
# MAGIC     
# MAGIC To check whether a substring exists in a string (to check if the year is a range or not), we need to use the `in` operator. See in the example below:

# COMMAND ----------

if "male" in "female":
    print("The substring was found.")
else:
    print("The substring was not found.")

# COMMAND ----------

if "love" in "loving":
    print("The substring was found.")
else:
    print("The substring was not found.")

# COMMAND ----------

# MAGIC %md
# MAGIC Second step, to split a string into two parts we need to use the `str.split()` method. This method can help us to split a string into a list of strings. See a quick example below:

# COMMAND ----------

year_in_range = "1995-1998"
print(year_in_range.split("-"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.1.7(HARD):
# MAGIC 
# MAGIC The `stripped_test_data` list, `strip_characters()` function and `bad_chars` list are provided for you in the editor below.
# MAGIC 
# MAGIC 1. Create a function called `process_date()` which accepts a string, and follows the logic we outlined above:
# MAGIC     - Checks if the dash character ``-`` is in the string so we know if it's a range or not.
# MAGIC     - If it is a range:
# MAGIC         * Splits the string into two strings, before and after the dash character.
# MAGIC         * Converts the two numbers to the integer type and then averages them by adding them together and dividing by two.
# MAGIC         * Uses the `round()` function to round the average, so values like 1964.5 become 1964.
# MAGIC     - If it isn't a range:
# MAGIC         - Converts the value to an `integer` type.
# MAGIC     - Finally, returns the value.
# MAGIC 2. Create an empty list `processed_test_data`.
# MAGIC 3. Loop over the `test_data` list using the `strip_characters()` function and your `process_date()` function. Process the dates and append each processed date back to the `processed_test_data` list.
# MAGIC 
# MAGIC 
# MAGIC 4. (OPTIONAL) Once your code works with the test data, you can then iterate over the moma list of lists. This list contains several date formats that we have not discussed so far. Try to deal with them, and any error you might get, in a way that seems sensible to you. In each iteration:
# MAGIC     - Create an empty list called `moma_dates`.
# MAGIC     - Loop over the rows of the `moma` list of lists.
# MAGIC     - Assign the value from the Date column (index `8`) to a variable.
# MAGIC     - Use the `strip_characters()` function to remove any bad characters.
# MAGIC     - Use the `process_date()` to convert the date.
# MAGIC     - Perform any other processing that you see fit to get a clean, single date.
# MAGIC     - Append the processed value to `moma_dates`.
# MAGIC 
# MAGIC  

# COMMAND ----------

from csv import reader
opened_file = open('Artworks.csv', encoding='utf8')
read_file = reader(opened_file)
moma = list(read_file)

test_data = ["1912", "1929", "1913-1923",
             "(1951)", "1994", "1934",
             "c. 1915", "1995", "c. 1912",
             "(1988)", "2002", "1957-1959",
             "c. 1955.", "c. 1970's", 
             "C. 1990-1999"]

def strip_characters(string):
    for char in string:
        if char not in '01234567890-':
            string = string.replace(char,"")
    return string



# Start your code here:

