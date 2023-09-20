# Databricks notebook source
# MAGIC %md
# MAGIC ## About This Notebook
# MAGIC 
# MAGIC Our focus in this notebook is twofold. Firstly, we focus on **understanding modules** and **ways to import them**. This is the more crucial part of this Notebook. Without this knowledge, we won't be able to move forward. Secondly, we will go into dates and times. Do not worry at all if you don't go too thoroughly through this part. You can always come back to this notebook and read through dates and times when you have the need.
# MAGIC 
# MAGIC The data from date/time contains a lot of information:
# MAGIC - Weather data with dates and/or times.
# MAGIC - Computer logs with the timestamp for each event.
# MAGIC - Sales data with date/time range included.
# MAGIC 
# MAGIC In this session, we will be working with records of visitors to the White House which was published in 2009.
# MAGIC ***

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Importing Modules (IMPORTANT)
# MAGIC 
# MAGIC In earlier notebooks, we used the csv module to make reading CSV files easier. 
# MAGIC > In Python, a **module** is simply a collection of variables, functions, and/or classes (which we'll collectively call 'definitions') that can be imported into a Python script.
# MAGIC 
# MAGIC **Python contains many standard modules** that help us perform various tasks, such as performing advanced mathematical operations, working with specific file formats and databases, and working with dates and times.
# MAGIC 
# MAGIC The **csv module** is one of the many standard modules from Python.
# MAGIC 
# MAGIC Whenever we use definitions from a module, we first need to import those definitions. There are a number of ways we can import modules and their definitions using the `import` statement. You can ready more about the `import` statement [here](https://docs.python.org/3/reference/simple_stmts.html#import). 
# MAGIC 
# MAGIC *Note: Please note that the cells below are formatted as raw text (not as a code). We do not want to create a mess by importing same module several times, in different ways.*
# MAGIC 
# MAGIC #### 1. Import the whole module by name. This is the most common method for importing a module.

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC # import the module
# MAGIC import csv
# MAGIC 
# MAGIC # definitions are available using the format
# MAGIC # module_name.definition_name
# MAGIC csv.reader()
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Import the whole module with an ``alias``. This is especially useful if a module name is long and we need to type it a lot.

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC # import the module with an alias
# MAGIC import csv as c
# MAGIC 
# MAGIC # definitions are available using the format
# MAGIC # alias.definition_name
# MAGIC c.reader()
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Import one or more definitions from the module by name. This is the technique we've used so far. This technique is useful if you want only a single or select definitions and don't want to import everything.

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC # import a single definition
# MAGIC from csv import reader
# MAGIC 
# MAGIC # the definition you imported is available
# MAGIC # by name
# MAGIC reader()
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC # import multiple definitions
# MAGIC from csv import reader, writer
# MAGIC 
# MAGIC # the definitions you imported are available
# MAGIC # using the format definition_name
# MAGIC reader()
# MAGIC writer()
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. Import all definitions with a wildcard. This is useful if you want to import and use many definitions from a module.

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC # import all definitions
# MAGIC from csv import *
# MAGIC 
# MAGIC # all definitions from the module are
# MAGIC # available using the format definition_name
# MAGIC reader()
# MAGIC writer()
# MAGIC get_dialect()``
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC Choosing which option to use when importing is often a matter of taste, but it's good to keep in mind how each choice can affect the readability of your code:
# MAGIC 
# MAGIC - If we're importing a long-name module by name and use it often, our code can become harder to read.
# MAGIC - If we use an uncommon alias, it may not be clear in our code which module we are using.
# MAGIC - If we use the specific definition or wildcard approach, and the script is long or complex, it may not be immediately clear where a definition comes from. This can also be a problem if we use this approach with multiple modules.
# MAGIC - If we use the specific definition or wildcard approach, it's easier to accidentally overwrite an imported definition.
# MAGIC 
# MAGIC In the end, there is often more than one "correct" way, so the most important thing is to be mindful of the trade-offs when you make a decision on how to import definitions from modules.
# MAGIC 
# MAGIC We'll learn about these trade-offs in the next screen as we learn about Python's datetime module, and make a decision on how to import it for our needs.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. The Datetime Module
# MAGIC 
# MAGIC There are three standard modules in Python that can help us working with dates and times.
# MAGIC - The calendar module
# MAGIC - The time module
# MAGIC - The datetime module
# MAGIC 
# MAGIC The module that we will go in deep into is the
# MAGIC [datetime module](https://docs.python.org/3/library/datetime.html#module-datetime). 
# MAGIC 
# MAGIC The datetime module contains a number of classes, including:
# MAGIC 
# MAGIC - `datetime.datetime`: For working with date and time data.
# MAGIC - `datetime.time`: For working with time data only.
# MAGIC - `datetime.timedelta`: For representing time periods.
# MAGIC 
# MAGIC You see that the first class, datetime, has the same name as the module. This could create confusion in our code. Now, let's look at different ways of importing and working with this first class, and the pros and cons.
# MAGIC 
# MAGIC <b>Import the whole module by name</b>
# MAGIC - Pro: It's super clear whenever you use datetime whether you're referring to the module or the class.
# MAGIC - Con: It has the potential to create long lines of code, which can be harder to read.
# MAGIC See example below:

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC # import the datetime module
# MAGIC import datetime
# MAGIC 
# MAGIC # use the datetime class
# MAGIC my_datetime_object = datetime.datetime()
# MAGIC # the first datetime represents the datetime module
# MAGIC # the second datetime represents the datetime class
# MAGIC 
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC <b>Import definitions via name or wildcard</b>
# MAGIC - Pro: Shorter lines of code, which are easier to read.
# MAGIC - Con: When we use datetime, it's not clear whether we are referring to the module or the class.
# MAGIC 
# MAGIC See Example below:

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC # import the datetime module
# MAGIC from datetime import datetime 
# MAGIC 
# MAGIC # import all definitions using wildcard
# MAGIC from datetime import *
# MAGIC 
# MAGIC # use the datetime class
# MAGIC my_datetime_object = datetime()
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC <b> Import whole module by alias </b>
# MAGIC - Pro: There is no ambiguity between dt (alias for the module) and dt.datetime (the class).
# MAGIC - Con: The dt alias isn't common convention, which would cause some confusion for other people reading our code.
# MAGIC See example below:

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC # import the datetime module 
# MAGIC import datetime as dt
# MAGIC 
# MAGIC # use the datetime class
# MAGIC my_datetime_object = dt.datetime()
# MAGIC 
# MAGIC # dt is the alias for the datetime module
# MAGIC # datetime() is the datetime class as we mentioned before
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.4.2:
# MAGIC Your exercise is to import the datetime module with the alias `dt`.

# COMMAND ----------

#Start your code below:


import datetime as dt

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The Datetime Class
# MAGIC 
# MAGIC The datetime.datetime class is the most commonly-used class from the datetime module, and has attributes and methods designed to work with data containing both the date and time. The signature of the class is below (with some lesser used parameters omitted):

# COMMAND ----------

# MAGIC %md
# MAGIC ````python 
# MAGIC datetime.datetime(year, month, day, hour=0, minute=0, second=0)
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC The above code indicates that the `year`, `month`, and `day` arguments are required. The time arguments are optional and can be set to the equivalent of midnight if omitted.
# MAGIC 
# MAGIC Now, let's take a look at an example of creating a datetime object.

# COMMAND ----------

# we'll import the datetime module and give it the alias dt
import datetime as dt

# we'll instantiate an object representing January 1, 2000
eg_1 = dt.datetime(2000, 1, 1)
print(eg_1)

# Let's instantiate a second object
# this time with both a date and a time
eg_2 = dt.datetime(1990, 4, 22, 21, 26, 2)
print(eg_2)

# COMMAND ----------

# MAGIC %md
# MAGIC This object represents 26 minutes and 2 seconds past 9 p.m. on the 22th of April, 1990.

# COMMAND ----------

# MAGIC %md
# MAGIC We can specify some but not all time arguments — in the following example we pass a value for hour and minute but not for second:

# COMMAND ----------

import datetime as dt

eg_3 = dt.datetime(1997, 10, 7, 9, 22)
print(eg_3)

# COMMAND ----------

# MAGIC %md
# MAGIC This object `eg_3` represents 9:22 a.m. on the 7th of October, 1997.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.4.3:
# MAGIC 
# MAGIC 1. Import the datetime class using the alias `dt`.
# MAGIC 2. Instantiate a datetime object representing midnight on June 16, 1911. Assign the object to the variable name `ibm_founded`.
# MAGIC 3. Instantiate a datetime object representing 8:17 p.m. on July 20, 1969. Assign the object to the variable name `man_on_moon`.

# COMMAND ----------

# Start your code below:

import datetime as dt
ibm_founded = dt.datetime(1911,6,16,0,0)
print(ibm_founded)

man_on_moon = dt.datetime(1969,7,20,8,17)
print(man_on_moon)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Using Strptime to Parse Strings as Dates

# COMMAND ----------

# MAGIC %md
# MAGIC Take a look at the code cell below. What we do there is trying to turn the date and time information stored in a string into a datetime object. It is a bit tricky as we need to use various methods to clean the string.

# COMMAND ----------

import datetime as dt

date_string = '12/18/15 16:39'

#Split date_string into two strings, either side of the space character
date,time = date_string.split()

#Split into string date components by their respective separators
hr, mn = time.split(':')
mnth, day, yr = date.split('/')

#Convert the string date components to integer components
hr = int(hr)
mn = int(mn)
mnth = int(mnth)
day = int(day)
yr = int(yr)

#Use the integer components to instantiate a datetime object
date_dt = dt.datetime(yr, mnth, day, hr, mn)

print(date_dt)
print(type(date_dt))

# COMMAND ----------

# MAGIC %md
# MAGIC We see that `datetime.strptime()` [constructor](https://docs.python.org/3/library/datetime.html#datetime.datetime.strptime) returns a datetime object. It defined the datetime object using a syntax system to describe date and time formats called `strftime`. (Pay attention to strftime with an "f" versus the constructor strptime with a "p".)
# MAGIC 
# MAGIC The strftime syntax consists of a `%` character followed by a single character which specifies a date or time part in a particular format.
# MAGIC 
# MAGIC For example "09/102/1998":

# COMMAND ----------

from datetime import datetime

datetime.strptime("09/02/1998", "%d/%m/%Y")

#%d - the day of the month in a two digit format, eg "09"
#%m - the month of the year in a two digit format, eg "02"
#%Y - the year in a four digit format, eg "1998"

# COMMAND ----------

# MAGIC %md
# MAGIC The first argument from the `datetime.strptime()` constructor is the string we want to parse, and the second argument is a string that helps us specify the format of the datetime object.
# MAGIC 
# MAGIC The `%d`, `%m`, and `%Y` format codes specify a two-digit day, two-digit month, and four-digit year respectively, and the forward slashes between them specify the forward slashes in the original string. Let's take a look at an example below:

# COMMAND ----------

date_1_str = "24/12/1984"
date_1_dt = dt.datetime.strptime(date_1_str, "%d/%m/%Y")
print(type(date_1_dt))
print(date_1_dt)

# COMMAND ----------

# MAGIC %md
# MAGIC Do you see that the constructor returns a datetime object?
# MAGIC 
# MAGIC Now, let's look at another example: "12-24-1984", the same exact date as above. What's different is the date parts are separated using a dash instead of a slash, in addition the order of the day and month are reversed:

# COMMAND ----------

date_2_str = "12-24-1984"
date_2_dt = dt.datetime.strptime(date_2_str, "%m-%d-%Y")
print(date_2_dt)

# %m - the month of the year in a wo digit format
# %d - the day of the month in a two digit format
# %Y - the year in a four digit format

# COMMAND ----------

# MAGIC %md
# MAGIC Below is a table of the most common format codes, take a look and simply know their existance. You don't need to known them by heart, you can always look them up in the [Python documentation](https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior).
# MAGIC 
# MAGIC |Strftime Code| Meaning| Examples|
# MAGIC |-|-|-|
# MAGIC |%d|Day of the month as a zero-padded number1|04|
# MAGIC |%A|Day of the week as a word2|Monday|
# MAGIC |%m|Month as a zero-padded number1|09|
# MAGIC |%Y|Year as a four-digit number|1901|
# MAGIC |%y|Year as a two-digit number with zero-padding1, 3|01 (2001), 88 (1988)|
# MAGIC |%B|Month as a word2|September|
# MAGIC |%H|Hour in 24 hour time as zero-padded number1|05 (5 a.m.),15 (3 p.m.)|
# MAGIC |%p|a.m. or p.m.2|AM|
# MAGIC |%I|Hour in 12 hour time as zero-padded number1|05 (5 a.m., or 5 p.m. if AM/PM indicates otherwise)|
# MAGIC |%M|Minute as a zero-padded number1|07|

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Using Strftime to format dates
# MAGIC 
# MAGIC Below is a list of attributes from the <b> datetime </b> class, which can help us retrieve the various parts that make up the date stored within the object much easier:
# MAGIC 
# MAGIC - `datetime.day:` The day of the month.
# MAGIC - `datetime.month:` The month of the year.
# MAGIC - `datetime.year:` The year.
# MAGIC - `datetime.hour:` The hour of the day.
# MAGIC - `datetime.minute:` The minute of the hour.
# MAGIC 
# MAGIC How can we use those attributes to extract the values? Look at the example below:

# COMMAND ----------

dt_object = dt.datetime(1984, 12, 24)

# We retrieve day value
day = dt_object.day 

# We retrieve month value
month = dt_object.month

# We retrieve year value
year = dt_object.year

# We use the retrieved value and form a new string
dt_string = "{}/{}/{}".format(day, month, year)
print(dt_string)
print(type(dt_string))

# COMMAND ----------

# MAGIC %md
# MAGIC It seems like what we performed above is a lot of code. There is a much easier method called [` datetime.strftime() `](https://docs.python.org/3/library/datetime.html#datetime.datetime.strftime), which will return a string representation of the date using the strftime syntax. Don't mix up strptime and strftime:
# MAGIC - strptime >> str-p-time >> string parse time
# MAGIC - strftime >> str-f-time >> string format time
# MAGIC 
# MAGIC With the `strftime()` method we can use `%d`, `%m`, and `%Y` to represent the date, month, and year.

# COMMAND ----------

dt_object = dt.datetime(1984, 12, 24)
dt_string = dt_object.strftime("%d/%m/%Y")
print(dt_string)

# COMMAND ----------

# MAGIC %md
# MAGIC Another way is, we can use `%B` to represent the month as a word:

# COMMAND ----------

dt_string = dt_object.strftime("%B %d, %Y")
print(dt_string)

# COMMAND ----------

# MAGIC %md
# MAGIC What else can we do? For a more granular representation of the time, we can use `%A`, `%I`, `%M`, and `%p` to represent the day of the week, the hour of the day, the minute of the hour, and a.m./p.m.:

# COMMAND ----------

dt_string = dt_object.strftime("%A %B %d at %I:%M %p")
print(dt_string)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. The Time Class
# MAGIC 
# MAGIC The time class holds only time data: hours, minutes, seconds, and microseconds.
# MAGIC An example to instantiate a time object is like this:

# COMMAND ----------

import datetime

datetime.time(hour=0, minute=0, second=0, microsecond=0)

# COMMAND ----------

# MAGIC %md
# MAGIC It is also possible to instantiate a time object without arguments. It will simply represent the time "0:00:00" (midnight). Otherwise, we can pass arguments for any or all of the hour, minute and second and microsecond parameters. Let's look at an example for the time 6:30 p.m.:

# COMMAND ----------

two_thirty = dt.time(18, 30)
print(two_thirty)

# COMMAND ----------

# MAGIC %md
# MAGIC Pay attention that we provided arguments in 24-hour time (an integer between 0 and 23). Let's look at an example of instantiating a time object for five seconds after 10 a.m.:

# COMMAND ----------

five_sec_after_10am = dt.time(10,0,5)
print(five_sec_after_10am)

# COMMAND ----------

# MAGIC %md
# MAGIC We can also create a time object from a datetime object using the `datetime.datetime.time()` method. This method returns a time object representing the time data from the datetime object.

# COMMAND ----------

# Version one
jfk_shot_dt = dt.datetime(1963, 11, 22, 12, 30)
print(jfk_shot_dt)

# COMMAND ----------

# Version two
jfk_shot_t = jfk_shot_dt.time()
print(jfk_shot_t)

# COMMAND ----------

# MAGIC %md
# MAGIC There is no `strptime()` constructor within the time class.  But if we need to parse times in string form, `datetime.strptime()` can be used and then converted directly to a time object like this:

# COMMAND ----------

time_str = "8:00"
time_dt = dt.datetime.strptime(time_str,"%H:%M")
print(time_dt)

# COMMAND ----------

time_t = time_dt.time()
print(time_t)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Comparing time objects
# MAGIC 
# MAGIC One of the best features of time objects is comparison. Take a look at the comparison example below:

# COMMAND ----------

t1 = dt.time(15, 30)
t2 = dt.time(10, 45)

comparison = t1 > t2
print(comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC There are also Python built-in functions like `min()` and `max()` that we can use for time class:

# COMMAND ----------

times = [
           dt.time(23, 30),
           dt.time(14, 45),
           dt.time(8, 0)
        ]

print(min(times))

# COMMAND ----------

print(max(times))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Calculations with Dates and Times
# MAGIC Just like time objects, datetime objects also support comparison operators like `>` and `<`. What about mathematical operators like `-` and `+`? Let's see if they work, too, starting with `+`:

# COMMAND ----------

dt1 = dt.datetime(2022, 4, 14)
dt2 = dt.datetime(2022, 3, 29)
print(dt1 + dt2)

# COMMAND ----------

# MAGIC %md
# MAGIC You see that when we  try to add two date objects using the `+` operator, we get a `TypeError` which tells us the operator is not valid.
# MAGIC 
# MAGIC But how about the <b> `-` </b> operator?

# COMMAND ----------

print(dt1 - dt2)

# COMMAND ----------

# MAGIC %md
# MAGIC It works! When we use the **`-`** operator with two date objects, the result is the time difference between the two datetime objects. 
# MAGIC 
# MAGIC Let's look at the type of the resulting object:

# COMMAND ----------

diff = dt1 - dt2
print(type(diff))

# COMMAND ----------

# MAGIC %md
# MAGIC The difference between two datetime objects is a `timedelta` object! Those take the following parameters:

# COMMAND ----------

datetime.timedelta(days=0, seconds=0, microseconds=0,
                   milliseconds=0, minutes=0, hours=0, weeks=0)

# COMMAND ----------

# MAGIC %md
# MAGIC Look carefully at the order of the arguments! It probably isn't the ordering you would expect. For this reason it can be clearer to use keyword arguments when instantiating objects if we are using anything other than days:

# COMMAND ----------

two_days = dt.timedelta(2)
print(two_days)

# COMMAND ----------

three_weeks = dt.timedelta(weeks=3)
print(three_weeks)

# COMMAND ----------

one_hr_ten_mins = dt.timedelta(hours=1, minutes=10)
print(one_hr_ten_mins)

# COMMAND ----------

# MAGIC %md
# MAGIC Lastly, timedelta objects can also be added or subtracted from datetime objects!

# COMMAND ----------

# we try to find the date one week from a date object
d1 = dt.date(1963, 2, 21)
d1_plus_1wk = d1 + dt.timedelta(weeks=1)
print(d1_plus_1wk)
