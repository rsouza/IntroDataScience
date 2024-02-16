# Databricks notebook source
# MAGIC %md
# MAGIC ## About This Notebook
# MAGIC
# MAGIC In this notebook we are going to dig into new topics. We will dig into logic and logical operators. In essence, when we code, we often want to control the *flow* of our code. For example, **IF** it is sunny, **THEN** go outside, **ELSE** stay inside. We are controlling the flow of what is happening with operators such as ``if``, ``or``, ``else``. We are going to discuss these three operators in this notebook.
# MAGIC ***

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. If Statements

# COMMAND ----------

from csv import reader

with open('../../../../Data/AppleStore.csv', encoding = 'utf8') as opened_file:
  read_file = reader(opened_file)
  apps_data = list(read_file)

# COMMAND ----------

# MAGIC %md
# MAGIC We have previously computed the average rating for all 7197 mobile apps. But there is more information for us to dig through. We might want to answer more granular questions like:
# MAGIC - What's the average rating of non-free apps?
# MAGIC - What's the average rating of free apps?
# MAGIC
# MAGIC In order to answer the two questions above, **we need to first separate free apps from non-free apps** and compute the average ratings just like we did in the previous chapter.
# MAGIC
# MAGIC Before we isolate the ratings of the free apps from the non-free apps, let's quickly learn another useful command for lists, namely `list_name.append()`.
# MAGIC See the code below:

# COMMAND ----------

number_list = [1,2,3]
number_list.append(4)
print(number_list)

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see, `number_list.append(4)` not only added the number 4 to the end our list, but it also modified the variable `number_list` in doing so, without having the reassign.
# MAGIC
# MAGIC Let's use this to create a new list containing only the ratings of the apps from `apps_data`. Note that we are iterating through `apps_data[1:]`, as the first row of `apps_data` (at index 0) contains the header of the dataset. 

# COMMAND ----------

from csv import reader

with open('../../../../Data/AppleStore.csv', encoding = 'utf8') as opened_file:
    read_file = reader(opened_file)
    apps_data = list(read_file)

ratings = []
for row in apps_data[1:]:
    rating = float(row[7])
    ratings.append(rating)
    
print(ratings[:5])

# COMMAND ----------

# MAGIC %md
# MAGIC In the code above, there is a problem with the ratings list. It includes both the ratings for the free and non-free apps. How do we only extract the ratings of the free apps? We can add a <b> condition </b> to our code above. 
# MAGIC To differentiate free apps from non-free apps, free apps always have a price which is equal to 0.0. 
# MAGIC
# MAGIC So if we implement a conditional statement where the price is equal to 0.0, then we will add this app to our ratings list. 
# MAGIC
# MAGIC We can achieve this by using an `if` statement like below:

# COMMAND ----------

ratings = []
for row in apps_data[1:]:  #we iterated over the apps_data[1:]
    rating = float(row[7]) #Assign the rating as a float to a variable named rating.
    price = float(row[4])  #Assign the price as a float to a variable named price.
    
    if price == 0.0: #If the price is equal to 0.0, that means this is a free app
        ratings.append(rating) # we append the rating to the ratings list

# COMMAND ----------

# MAGIC %md
# MAGIC In the code above, we have learned about the basics of the `if` statement.
# MAGIC - An if statement starts with `if` and it ends with `:` (just like the for loop)
# MAGIC - We use the `==` operator to check whether the price is equal to 0.0. Pay attention that `==` is not the same with `=`. (`=` is a variable assignment operator)
# MAGIC - `ratings.append(rating)` will only be executed if the if statement is true.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.4.1:
# MAGIC Complete the code in the editor to find the average rating for free apps.
# MAGIC
# MAGIC 1. Inside the for loop:
# MAGIC 2. Assign the price of an app as a float to a variable named ``price``. The price is the fifth element in each row (don't forget that the index starts at 0).
# MAGIC 3. If `price == 0.0`, append the value stored in rating to the `free_apps_ratings` list using the `list_name.append()` command (note the `free_apps_ratings` is already defined in the code editor). Be careful with indentation.
# MAGIC 4. Outside the for loop body, compute the average rating of free apps. Assign the result to a variable named ``avg_rating_free``. The ratings are stored in the `free_apps_ratings` list. Hint: sum(...) / len(...)

# COMMAND ----------

## Start your code below:
from csv import reader

with open('../../../../Data/AppleStore.csv', encoding = 'utf8') as opened_file:
    read_file = reader(opened_file)
    apps_data = list(read_file)

free_apps_ratings = []
for row in apps_data[1:]:
    rating = float(row[7])
    # Complete the code from here
    price = float(row[4])
    if price == 0.0:
        free_apps_ratings.append(rating)

avg_rating_free = sum(free_apps_ratings) / len(free_apps_ratings)

print(avg_rating_free)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Booleans
# MAGIC
# MAGIC In the previous function, we have used if `price == 0.0` to check whether price is equal to 0.0.
# MAGIC When we use the `==` operator to determine if two values are equal or not, the output returned will always be `True` or `False`.

# COMMAND ----------

print(10 == 10) #It's true that 10 is equal to 10
print(10 == 2)  #It's false that 10 is equal to 2

# COMMAND ----------

# MAGIC %md
# MAGIC Do you know the type of True and False?
# MAGIC They are **booleans**.

# COMMAND ----------

type(True)

# COMMAND ----------

type(False)

# COMMAND ----------

# MAGIC %md
# MAGIC Boolean values, which are True and False, are very important to any if statement.
# MAGIC An if statement must always be followed by one of the following:
# MAGIC
# MAGIC 1. a Boolean value, or
# MAGIC 2. an expression that evaluates to a Boolean value.
# MAGIC
# MAGIC So it always has something to do with a Boolean value.

# COMMAND ----------

#1. A Boolean value.

if True:
    print(10)

# COMMAND ----------

#2. An expression that evaluates to a Boolean value.
if 10 == 10:
    print(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Pay attention: if you evaluate the expression with just one ``=``, you will get an error like this:

# COMMAND ----------

# = is not the same as ==, a SyntaxError will be thrown
if 10 = 10:
    print(10)

# COMMAND ----------

# MAGIC %md
# MAGIC The indented code is only executed when if is followed by True. The code body will no be executed if the if is followed by False. See the example below:

# COMMAND ----------

if True:
    print('This boolean is true.')

if False:
    print('This code will not be executed.')

if True:
    print('The code above was not executed because "if" was followed by "False".')

# COMMAND ----------

# MAGIC %md
# MAGIC Note that we can write as much code as we want after the if statement like this:

# COMMAND ----------

if True:
    print(10)
    print(100)
    print(1000)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.4.2:
# MAGIC
# MAGIC In the code editor, we've already initialized the variable <b> price </b> with a value of 0.
# MAGIC Write the code so it matches the following:
# MAGIC 1. If price is equal to 0, print the string 'This is free'
# MAGIC 2. If price is equal to 1, print the string 'This is not free'

# COMMAND ----------

price = 0

#start your code here:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The Average Rating of Non-free Apps
# MAGIC
# MAGIC In the diagram below, we created a list of lists named `app_and_price`, and we want to extract the names of the free apps in a separate list. To do that, we:

# COMMAND ----------

app_and_price = [['Facebook', 0], ['Instagram', 0], ['Plants vs. Zombies', 0.99], 
                 ['Minecraft: Pocket Edition', 6.99], ['Temple Run', 0],
                 ['Plague Inc', 0.99]]

#create an empty list named free_apps
free_apps = []

#iterate over app_and_price
for app in app_and_price: # for each iteration, we:
    name = app[0]         # extract the name of the app and assign it to name
    price = app[1]        # extract the price of the app and assign it to price 
    
    if price == 0: # if the price of the app is equal to 0
        free_apps.append(name) # append the name of the app to free_apps
        
print(free_apps)

# COMMAND ----------

# MAGIC %md
# MAGIC Do you remember a very similar example that we did at the beginning of this chapter?

# COMMAND ----------

#Import data set
from csv import reader

with open('../../../../Data/AppleStore.csv', encoding = 'utf8') as opened_file:
    read_file = reader(opened_file)
    apps_data = list(read_file)

ratings = []
for row in apps_data[1:]:  # we looped through a list of lists named apps_data
    rating = float(row[7]) # extract the rating of the app and assign the rating as a float to a variable named rating.
    price = float(row[4])  # extract the price of the app and assign the price as a float to a variable named price.
    
    if price == 0.0: # If the price is equal to 0.0, that means this is a free app
        ratings.append(rating) # we append the rating to the ratings list

# COMMAND ----------

# MAGIC %md
# MAGIC These two functions basically do the same task. To extract free apps from a mixed list of free and non-free apps.
# MAGIC
# MAGIC When we want to extract the free apps from the list, we used the condition "if the price is equal to 0.0" (`if price == 0.0`).
# MAGIC How do we extract the non-free apps then? The "is not equal to" operator comes in the place. What we have done so far is by using the "is equal to" operator which is just the same as `==`. And if we want to extract the non-free apps, we can use the "is not equal to" operator which is just like this: `!=`.
# MAGIC See the example below:

# COMMAND ----------

print(10 != 0)
print(10 != 10)

# COMMAND ----------

# MAGIC %md
# MAGIC How can we use the `!=` operator for our price variable then? It could look as follows:

# COMMAND ----------

price = 10

print(price != 0)
print(price != 10)

if price != 0:
    print('Not free')
    
if price != 10:
    print('Price is not equal to 10')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.4.3 (OPTIONAL): 
# MAGIC
# MAGIC Can you modify the code in the editor below to compute ``the average rating of non-free apps``?
# MAGIC
# MAGIC 1. Change the name of the empty list from `free_apps_ratings` to `non_free_apps_raings`.
# MAGIC 2. Change the conditional statement `if price == 0.0` to extract the ratings of non-free apps.
# MAGIC 3. Change `free_apps_ratings.append(rating)` and append to the new list `non_free_apps_ratings`.
# MAGIC 4. Compute the average value by summing up the values in `non_free_apps_ratings` and dividing by the length of this list. Assign the result to ``avg_rating_non_free``.

# COMMAND ----------

# INITIAL CODE
from csv import reader

with open('../../../../Data/AppleStore.csv', encoding = 'utf8') as opened_file:
    read_file = reader(opened_file)
    apps_data = list(read_file)

free_apps_ratings = []
for row in apps_data[1:]:
    rating = float(row[7])
    price = float(row[4])   
    if price == 0.0:
        free_apps_ratings.append(rating)
    
avg_rating_free = sum(free_apps_ratings) / len(free_apps_ratings)

print(avg_rating_free)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. The Average Rating of Gaming Apps
# MAGIC We have learned about the ``==`` and ``!=`` operators, and have only worked with integers and floats. Do you know we can use these two operators with other data types as well? 

# COMMAND ----------

# operators with strings
print('Games' == 'Music')
print('Games' != 'Music')

# COMMAND ----------

# operators with lists
print([1,2,3] == [1,2,3])
print([1,2,3] == [1,2,3,4])

# COMMAND ----------

# MAGIC %md
# MAGIC With such functionalities, we can answer more detailed questions about our data set like what the average rating of gaming and non-gaming apps is.
# MAGIC
# MAGIC We can see that from the data set that the ``prime_genre`` column describes the app genre. If the app is a gaming app, the genre of the app will be encoded as 'Games'.
# MAGIC
# MAGIC To compute the average rating of gaming apps, we can use the same approach as we took in the previous screen when we computed the average rating of free and non-free apps.

# COMMAND ----------

games_ratings = [] #initialize an empty list named games_ratings

for row in apps_data[1:]: 
    rating = float(row[7]) # extract the rating information and assign the rating to the variable rating
    genre = row[11] # extract the genre information and assign the genre to the variable named genre
    
    if genre == 'Games': # if the genre is 'Games'
        games_ratings.append(rating) # append the rating value stored in rating to the list games_ratings

avg_ratings_games = sum(games_ratings) / len(games_ratings) #compute the average rating of gaming apps
print(avg_ratings_games)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.4.4:
# MAGIC With the code above, can you use the same techniques to compute the average rating of ``non-gaming apps``?

# COMMAND ----------

# reading the dataset
from csv import reader

with open('../../../../Data/AppleStore.csv', encoding = 'utf8') as opened_file:
  read_file = reader(opened_file)
  apps_data = list(read_file)

# start your code here:

# Initialize an empty list named non_games_ratings.

# Loop through the apps_data


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Multiple Conditions
# MAGIC ### 5.1 The `and` operator
# MAGIC
# MAGIC So far, we've only worked with single conditions, like:
# MAGIC
# MAGIC - If price equals 0.0 (`if price == 0`)
# MAGIC - If genre equals "Games" (`if genre == 'Games'`)
# MAGIC
# MAGIC However, single conditions won't help us answering some more complicated questions, like:
# MAGIC
# MAGIC - What's the average rating of free gaming apps?
# MAGIC - What's the average rating of non-free gaming apps?
# MAGIC - What's the average rating of free non-gaming apps?
# MAGIC - What's the average rating of non-free non-gaming apps?
# MAGIC
# MAGIC To solve this, do you know that we can combine two or more conditions together into a single ``if`` statement using the ``and `` keyword just like the English language.

# COMMAND ----------

app1_price = 0
app1_genre = 'Games'

if app1_price == 0 and app1_genre == 'Games':
    print('This is a free game!')
    
print(app1_price == 0 and app1_genre == 'Games')

# COMMAND ----------

app1_price = 19
app1_genre = 'Games'

if app1_price == 0 and app1_genre == 'Games':
    print('This is a free game!')
    
print(app1_price == 0 and app1_genre == 'Games')

# COMMAND ----------

# MAGIC %md
# MAGIC Do you see that the code `app1_price == 0 and app1_genre == 'Games'` outputs a single Boolean value?
# MAGIC Python evaluate any combination of Booleans to a single boolean value:

# COMMAND ----------

print(True and True)
print(True and False)
print(False and True)
print(False and False)

# COMMAND ----------

# MAGIC %md
# MAGIC The rule is that when we combine Booleans using ``and``, the end output ``True`` can only take place if expressions within are all ``True``. If there is any ``False`` within, then the end output can only be ``False``. 
# MAGIC
# MAGIC You might recall the ``truth table`` of the AND operator from your university class, which explains exactly how the ``and`` operator functions in Python.  It looks something like this:
# MAGIC
# MAGIC <img src="https://www.chilimath.com/wp-content/uploads/2020/02/truth-table-conjunction-792x1024.gif" width="200" height="200" />

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 The or Operator
# MAGIC
# MAGIC When we look at the `prime_genre` column we can see that there are the genres "Social Networking" and "Games".
# MAGIC
# MAGIC As a data scientist, sometimes we might want to figure out the average rating of both categories. That means we need to isolate the ratings of all the apps whose genre is either "Social Networking" or "Games" into a separate list, and then calculate the average value in the same way as we have always done until now.
# MAGIC
# MAGIC How do we isolate the ratings of these apps? This time, we can't use the `and` operator. We need an `or` operator, because it doesn't make any sense to use the `and` operator like this:
# MAGIC <br>`if genre == 'Social Networking' and genre == 'Games'`
# MAGIC
# MAGIC Each app has only one genre in the `prime_genre` column, **you won't have any apps with both social networking and games as its genre**.
# MAGIC This is where the ``or`` operator comes into place. Look at the code below:

# COMMAND ----------

games_social_ratings = []

for row in apps_data[1:]:
    ratings = float(row[7])
    genre = row[11]
    
    if genre == 'Social Networking' or genre == 'Games':
        games_social_ratings.append(rating)

print(games_social_ratings[:5])
len(games_social_ratings)

# COMMAND ----------

# MAGIC %md
# MAGIC Do know then know how `or` operator behaves in Python? See below:

# COMMAND ----------

# The OR operator
print(True or True)
print(True or False)
print(False or True)
print(False or False)

# COMMAND ----------

# MAGIC %md
# MAGIC Again the results from the code above can be summarized in a ``truth table``. It explains exactly how the ``or`` operator functions in Python.  
# MAGIC
# MAGIC | A     | B     | A ``or`` B |
# MAGIC | ----- | ----- | ---------- |
# MAGIC | True  | True  | True       |
# MAGIC | True  | False | True       |
# MAGIC | False | True  | True       |
# MAGIC | False | False | False      |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Let's compare it again to the `and` operator:

# COMMAND ----------

# The AND operator
print(True and True)
print(True and False)
print(False and True)
print(False and False)

# COMMAND ----------

# MAGIC %md
# MAGIC You see that the ``and`` and the ``or`` operators behave differently. The `or` operator will evaluate the statement to ``True`` if **any** of the Booleans are ``True``:

# COMMAND ----------

print(False or False or False)
print(False or False or False or True)

# COMMAND ----------

# MAGIC %md
# MAGIC Returning to our apps example, the condition `if genre == 'Social Networking' or genre == 'Games'` will only resolve to `False` when an app's genre is neither "Social Networking" nor "Games." Otherwise, it will resolve to True.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 The ``not`` operator (OPTIONAL):
# MAGIC
# MAGIC Another usefull keyword when working with conditional statements and logigal operators is the `not` keyword.
# MAGIC It can be used to negate the negate value of a boolean. In other words it is an operator that returns `True` if the statement is not `True`.
# MAGIC
# MAGIC | A     | ``not`` A |
# MAGIC |------ | --------- |
# MAGIC | True  | False     |
# MAGIC | False | True      |
# MAGIC
# MAGIC Let's have a look at an example:

# COMMAND ----------

keep_the_secret = True

if not keep_the_secret:
  print("This is the secret")

# COMMAND ----------

# MAGIC %md
# MAGIC Of course the ``not`` operator works in conjunciton with the previously introduced operators:

# COMMAND ----------

first_bool = True
second_bool = False

if first_bool and not second_bool:
  print("This expression equates 'True'!")

if not first_bool or second_bool:
  print("This expression equates 'False'!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Combining Logical Operators
# MAGIC
# MAGIC In the previous exercise, the average rating of the apps whose genre is either "Social Networking" or "Games." is computed. Now we can ask even more specific questions, like:
# MAGIC
# MAGIC What is the average rating of <b> free </b> apps whose genre is either "Social Networking" or "Games"?
# MAGIC What is the average rating of <b> non-free </b> apps whose genre is either "Social Networking" or "Games"?
# MAGIC
# MAGIC To answer the first question, we need to isolate the apps that:
# MAGIC
# MAGIC - are in either the "Social Networking" or "Games" genre, and
# MAGIC - have a price of 0.0.
# MAGIC
# MAGIC To isolate these apps, we can combine `or` with `and` in a single if statement:

# COMMAND ----------

free_games_social_ratings = []

for row in apps_data[1:]:
    rating = float(row[7])
    genre = row[11]
    price = float(row[4])
    
    if(genre == 'Social Networking' or genre == 'Games') and price == 0:
        free_games_social_ratings.append(rating)

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that we enclosed the `genre == 'Social Networking' or genre == 'Games'` part within parentheses. This helps Python understand the specific logic we want for our if statement.
# MAGIC If the parentheses are not applied correctly, Python will lead us to unwanted results.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Attention
# MAGIC Note that
# MAGIC
# MAGIC ````python
# MAGIC if (genre == 'Social Networking' or genre == 'Games') and price == 0:
# MAGIC
# MAGIC ````
# MAGIC is not the same as
# MAGIC ````python
# MAGIC if genre == 'Social Networking' or (genre == 'Games' and price == 0):
# MAGIC ````

# COMMAND ----------

# No parentheses
print(True or False and False)

# COMMAND ----------

# MAGIC %md
# MAGIC Now please observe the code below:

# COMMAND ----------

app_genre = 'Social Networking'
app_price = 100 # This is a non-free app

if app_genre == 'Social Networking' or app_genre == 'Games' and app_price == 0:
    print('This gaming or social networking app is free!!')

# COMMAND ----------

# MAGIC %md
# MAGIC You see above that the code above has a <b> logical error </b>. We have printed 'This gaming or social networking app is free!!' for a non-free app. 
# MAGIC
# MAGIC If we place the parentheses corretly, it makes a huge difference:

# COMMAND ----------

# Correctly placed parentheses
print((True or False) and False)

# COMMAND ----------

app_genre = 'Social Networking'
app_price = 100 #This is a non-free app


if (app_genre == 'Social Networking' or app_genre == 'Games') and app_price == 0:
    print('This gaming or social networking app is free!!')

# COMMAND ----------

# MAGIC %md
# MAGIC We see that now this expression is evaluated correctly. Now you see the importance of using parentheses correctly. So make sure you place your parentheses correctly!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.4.6:
# MAGIC
# MAGIC Modify the code below to compute also the average rating of non-free apps whose genre is either ``Social Networking`` or ``Games``. Assign the result to a variable named ``avg_non_free``, print it and compare to `avg_free`.

# COMMAND ----------

from csv import reader

with open('../../../../Data/AppleStore.csv', encoding = 'utf8') as opened_file:
    read_file = reader(opened_file)
    apps_data = list(read_file)

free_games_social_ratings = []
non_free_ratings = []

# edit the code here:
for row in apps_data[1:]:
    rating = float(row[7])
    genre = row[11]
    price = float(row[4])
    
    if (genre == 'Social Networking' or genre == 'Games') and price == 0:
        free_games_social_ratings.append(rating)
    
avg_free = sum(free_games_social_ratings) / len(free_games_social_ratings)

print(avg_free)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Comparison Operators
# MAGIC
# MAGIC Did you remember the equal to and non-equal to operators that we learned before? We used the `==` and `!=` operators to check whether two values are equal or not. These two operators are called `comparison operators`.
# MAGIC
# MAGIC Please see the <b>comparison operator table</b> below:
# MAGIC
# MAGIC | Comparison (text) |  Comparison operator |  Comparison (code)| 
# MAGIC |------------|:------:|----------:| 
# MAGIC | A is equal to B| == | A==B |
# MAGIC | A is not equal to B |   != |   A !=B |
# MAGIC | A is greater than B | >|  A > B |
# MAGIC | A is greater than or equal to B |   >=  |   A >= B |
# MAGIC |A is less than B|<| A < B| 
# MAGIC |A is less than or equal to B| < =| A <= B|
# MAGIC
# MAGIC What does it look when we put comparison operators into actual code?

# COMMAND ----------

print(10 > 2)
print( 10 < 2)
print (50 >=50)
print( 30 <=15)

# COMMAND ----------

# MAGIC %md
# MAGIC Because a comparison operator will evaluate the expression and give us a single Boolean value as the final output, we can therefore also use such comparison operators inside the ``if`` statement. 
# MAGIC Like this:

# COMMAND ----------

app_name = 'Ulysses'
app_price = 24.99

if app_price > 20:
    print('This app is expensive!')
    
print(app_price > 20)

# COMMAND ----------

# MAGIC %md
# MAGIC These new comparison operators open up new possibilities for us to answer more granular questions about our data set like:
# MAGIC
# MAGIC - How many apps have a rating of 4 or greater?
# MAGIC - What is the average rating of the apps that have a price greater than 9 Dollars?
# MAGIC - How many apps have a price greater than 9 Dollars?
# MAGIC - How many apps have a price smaller than or equal to 9 Dollars?

# COMMAND ----------

# MAGIC %md
# MAGIC To answer the first question, we can write the code like this:

# COMMAND ----------

apps_4_or_greater = [] # initialized an empty list named apps_4_or_greater

for row in apps_data[1:]:
    rating = float(row[7]) # Stored the rating value as a float to a variable named rating
    if rating >= 4.0:
        apps_4_or_greater.append(rating)
        
len(apps_4_or_greater)

# COMMAND ----------

# MAGIC %md
# MAGIC Alternatively, we do not have to use ``append`` in this case:

# COMMAND ----------

n_of_apps = 0 #Initialized a variable n_of_apps with a value of 0

for row in apps_data[1:]:
    rating = float(row[7]) #Stored the rating value as a float to a variable named rating
    if rating >= 4.0:
        n_of_apps = n_of_apps + 1 #Incremented the value of n_of_apps by 1 
        #if the value stored in rating was greater than or equal to 4.0
        
print(n_of_apps)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. The else Clause
# MAGIC
# MAGIC Let's say we need to use information from the price column to label each app as ``free`` or ``non-free``. If the price is equal to ``0.0``, we want to label the app ``free``. Otherwise, we want to label it ``non-free``. To summarize this is Python language it's like this:

# COMMAND ----------

apps_data = [['GoodNotes', 19.99], ['Amazon', 0.0], ['Chrome', 0.0], ['Snapchat', 0.0]]

for app in apps_data:
    price = app[1] #We saved the price value to a variable named price.
    
    if price == 0.0: #If price == 0.0
        app.append('free') #we appended the string 'free' to the list app 
    if price != 0.0: #If price != 0.0
        app.append('non-free') #we appended the string 'non-free' to the list 
        
print(apps_data)

# COMMAND ----------

# MAGIC %md
# MAGIC For each iteration, the computer evaluate two expressons: ``price == 0.0`` and ``price != 0.0``. But once we know that ``price == 0.0`` for an app, it's redundant to also check `price != 0.0` for the same app — if we know that the price is ``0``, it doesn't make logical sense to also check whether the price is different than `0`.
# MAGIC
# MAGIC In our small data set above, we have 3 free apps. The computer evaluated ``price == 0.0`` as ``True`` 3 times, and then checked whether ``price != 0.0`` for the same number of times. There are 3 **redundant** operations performed in this small dataset. 
# MAGIC
# MAGIC However, if we have a data set with 5,000 free apps, we wouldn't want our computer to perform 5,000 redundant operations. This can be avoided using the if statement with an ``else`` clause:

# COMMAND ----------

apps_data = [['GoodNotes', 19.99], ['Amazon', 0.0], ['Chrome', 0.0], ['Snapchat', 0.0]]

for app in apps_data:
    price = app[1] # We saved the price value to a variable named price.
    
    if price == 0.0: # If price == 0.0
        app.append('free') # we appended the string 'free' to the list app 
    else: # If price  is not 0.0,
        app.append('non-free') # we appended the string 'non-free' to the list 
        
print(apps_data)

# COMMAND ----------

# MAGIC %md
# MAGIC You see that the two chunks of code above give the same result, but the latter ist much more *sufficient* and *sophisticated*.
# MAGIC
# MAGIC Pay attention that the code within the body of an else clause is executed only if the if statement that precedes it resolves to `False`.

# COMMAND ----------

# MAGIC %md
# MAGIC ~~~python
# MAGIC if False:
# MAGIC     print(10)
# MAGIC else:
# MAGIC     print(100)
# MAGIC ~~~
# MAGIC
# MAGIC Just like in our apps example:
# MAGIC ````python
# MAGIC price = 5
# MAGIC
# MAGIC if price == 0:
# MAGIC     print('free')
# MAGIC else: 
# MAGIC     print('not free')
# MAGIC ````
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ````python
# MAGIC price = 0
# MAGIC
# MAGIC if price == 0:
# MAGIC     print('free')
# MAGIC else: 
# MAGIC     print('not free')
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC In the apps example above, the code within the body of the ``else`` clause is executed only if ``price == 0.0`` evaluates to `False`. 
# MAGIC
# MAGIC If `price == 0.0` is `True`, then the line
# MAGIC
# MAGIC ````python
# MAGIC app.append('free') 
# MAGIC ````
# MAGIC is executed, and computer moves forward *without* executing the else clause.
# MAGIC
# MAGIC Note that an else clause must be combined with a preceding if statement. We can have an if statement without an else clause, but we can't have an else clause without a preceding if statement. In the example below, the else clause alone throws out a ``SyntaxError``:

# COMMAND ----------

else:
    print('This will not be printed, becausewe cannot have an else clause without a preceding if statement.' )

# COMMAND ----------

# MAGIC %md
# MAGIC When we combine a statement with a clause, we create a compound statement — combining an if statement with an else clause makes up a compound statement.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. The elif Clause
# MAGIC
# MAGIC What happens when we have a more granular label rather than just using "free" and "non-free" like below:
# MAGIC
# MAGIC | price | label |  
# MAGIC |------------|:------:|
# MAGIC | 0 | free|
# MAGIC | >0 - < 20 |  affordable |
# MAGIC | 20 - 50 | expensive |
# MAGIC |> 50| very expensive |

# COMMAND ----------

# MAGIC %md
# MAGIC Using what we learned from the previous section, our code will like below:

# COMMAND ----------

apps_data = [['GoodNotes', 19.99], ['Call of Duty Zombies', 5.0], ['Notability', 29.99], ['Snapchat', 0.0]]

for app in apps_data:
    price = app[1] 
    
    if price == 0.0: 
        app.append('free') 
    if price > 0.0 and price < 20:
        app.append('affordable')
    if price >= 20 and price < 50:
        app.append('expensive')
    if price >= 50:
        app.append('very expensive')
        
print(apps_data)

# COMMAND ----------

# MAGIC %md
# MAGIC When an app is free, ``price == 0.0`` evaluates to ``True`` and ``app.append('free')`` is executed. But then the computer continues to do redundant operations — it checks whether:
# MAGIC
# MAGIC -  `price > 0 and price < 20`
# MAGIC -  `price >= 20 and price < 50`
# MAGIC -  `price >= 50`
# MAGIC
# MAGIC We already know the three conditions above will evaluate to False once we find out that `price == 0.0` is `True`. To stop the computer from doing redundant operations, we can use **elif clauses**:

# COMMAND ----------

apps_data = [['GoodNotes', 19.99], ['Call of Duty Zombies', 5.0], ['Notability', 29.99], ['Snapchat', 0.0]]

for app in apps_data:
    price = app[1] 
    
    if price == 0.0: 
        app.append('free') 
    elif price > 0.0 and price < 20:
        app.append('affordable')
    elif price >= 20 and price < 50:
        app.append('expensive')
    elif price >= 50:
        app.append('very expensive')
        
print(apps_data)

# COMMAND ----------

# MAGIC %md
# MAGIC The code within the body of an elif clause is executed only if:
# MAGIC
# MAGIC - The preceding if statement (or the other preceding elif clauses) resolves to `False`; and
# MAGIC - The condition specified after the elif keyword evaluates to `True`.
# MAGIC
# MAGIC For example, if `price == 0.0` is evaluated to be `True`, the computer will execute `app.append('free')` and jump out of this section of code (skip the rest of the elif clauses).
# MAGIC
# MAGIC Notice that if we replace the last `elif` with `else` instead, the statement app.append('very expensive') will be executed even if the price has a value of -5 or -100 like this:

# COMMAND ----------

apps_data = [['GoodNotes', -19.99], ['Call of Duty Zombies', -5.0], ['Notability', 29.99], ['Snapchat', 0.0]]

for app in apps_data:
    price = app[1] 
    
    if price == 0.0: 
        app.append('free') 
    elif price > 0.0 and price < 20:
        app.append('affordable')
    elif price >= 20 and price < 50:
        app.append('expensive')
    else:
        app.append('very expensive')
        
print(apps_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's have some practice with the `else` and `elif` statements.
# MAGIC ### Task 1.4.9 (OPTIONAL):
# MAGIC
# MAGIC 1. Complete the code in the editor to label each app as ``free``, ``affordable``, ``expensive``, or ``very expensive``. Inside the loop:
# MAGIC   - If the price of the app is 0, label the app as ``free`` by appending the string ``free`` to the current iteration variable.
# MAGIC   - If the price of the app is greater than 0 and less than 20, label the app as ``affordable``. For efficiency purposes, use an elif clause.
# MAGIC   - If the price of the app is greater or equal to 20 and less than 50, label the app as ``expensive``. For efficiency purposes, use an elif clause.
# MAGIC   - If the price of the app is greater or equal to 50, label the app as ``very expensive``. For efficiency purposes, use an elif clause.
# MAGIC 2. Name the newly created column ``price_label`` by appending the string ``price_label`` to the first row of the ``apps_data`` data set.
# MAGIC 3. Inspect the header row and the first five rows to see some of the changes you made.

# COMMAND ----------

# INITIAL CODE
from csv import reader

with open('../../../../Data/AppleStore.csv', encoding = 'utf8') as opened_file:
    read_file = reader(opened_file)
    apps_data = list(read_file)

for app in apps_data[1:]:
    price = float(app[4])
    # Complete code from here
    

# COMMAND ----------

# MAGIC %md
# MAGIC In this chapter, we learned to combine conditional statements and clauses (`if`, `else`, `elif`) with logical operators (`and`, `or`) and comparison operators (`==`, `!=`, `>`, `>=`, `<`, `<=`) to answer some granular and more complicated questions.
