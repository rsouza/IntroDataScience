# Databricks notebook source
# MAGIC %md
# MAGIC ## About This Notebook
# MAGIC 
# MAGIC In this notebook, we are moving towards a very important new concept called **dictionary**. We already know how to work with lists and lists of lists (which can store data sets). These are however rather basic forms of data. Dictionaries offer us a key/value approach to store data. For example, we might have looped through a list of lists and extracted certain *information* into a dictionary. 
# MAGIC ***
# MAGIC 
# MAGIC ## 1. Dictionaries

# COMMAND ----------

# MAGIC %md
# MAGIC In the previous notebook we worked with the `AppleStore.csv` data set. The `cont_rating` column provides a lot of information regarding the content rating of each app. See below:<br>
# MAGIC 
# MAGIC |Content rating |Number of apps|
# MAGIC |--|--|
# MAGIC |4+|4,433|
# MAGIC |9+|987|
# MAGIC |12+|1,155|
# MAGIC |17+|622|

# COMMAND ----------

# MAGIC %md
# MAGIC How can we store the data above? We can do it in two ways:
# MAGIC - Using two separate lists
# MAGIC - Using a single list of lists

# COMMAND ----------

# Two lists
content_ratings = ['4+', '9+', '12+', '17+']
numbers = [4433, 987, 1155, 622]

# A list of lists
content_rating_numbers = [['4+', '9+', '12+', '17+'], [4433, 987, 1155, 622]]

# COMMAND ----------

# MAGIC %md
# MAGIC Looking at the lists above, you may be confused which content rating corresponds to which number. Each list element has an index number and we will transform the index numbers to content rating values. We can do this using a <b> dictionary </b> like this:

# COMMAND ----------

content_ratings = {'4+':4433, '9+':987, '12+':1155, '17+':622}
print(content_ratings)

# COMMAND ----------

# MAGIC %md
# MAGIC What we have done above is:
# MAGIC - Mapped each content rating to its corresponding number by following an `index:value` pattern. 
# MAGIC - Separated each pair with a comma
# MAGIC - Surrounded the sequence with curly braces {}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Indexing (IMPORTANT)
# MAGIC 
# MAGIC We can change the index numbers of a list to content rating values in a dictionary. So the connection between content ratings their corresponding numbers becomes much clearer.
# MAGIC 
# MAGIC Now the question arises of how we can retrieve the individual values of the content_ratings dictionary.  We can use indices like we did with the individual list elements following a `variable_name[index]` pattern:

# COMMAND ----------

content_ratings = {'4+':4433, '9+':987, '12+':1155, '17+':622}
print(content_ratings['4+'])
print(content_ratings['12+'])

# COMMAND ----------

# MAGIC %md
# MAGIC The indexing of dictionary is something different than the list. 
# MAGIC > The order of the dictionary is not necessarily preserved whereas in the lists, the order is **always** preserved.
# MAGIC 
# MAGIC For example, the index value 0 always retrieves the list element that's positioned first in a list. With dictionaries, there is no direct connection between the index of a value and the position of that value in the dictionary. That means that the order is unimportant. The index value '4+' will retrieve tha value 4433 no matter its position. It could be the first element, or the last element in the dictionary, is doesn't matter.

# COMMAND ----------

content_ratings = {'4+':4433, '9+':987, '12+':1155, '17+':622}
print(content_ratings)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.5.2:
# MAGIC 
# MAGIC 1. Retrieve values from the ``content_ratings`` dictionary.
# MAGIC     - Assign the value at index `'9+'` to a variable named `over_9`.
# MAGIC     - Assign the value at index `'17+'` to a variable named `over_17`.
# MAGIC 2. Print `over_9` and `over_17`.

# COMMAND ----------

### Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Alternative way of Creating a Dictionary
# MAGIC 
# MAGIC There is an alternative way of creating a dictionary like this:
# MAGIC 1. Create an empty dictionary.
# MAGIC 2. Add values one by one to that empty dictionary.
# MAGIC     - like this: `dictionary_name[index] = value`
# MAGIC     
# MAGIC Take for example, if we want to add a value 4455 with an index `'5+'` to a dictionary named `content_ratings`, we need to use the code:
# MAGIC 
# MAGIC ````python
# MAGIC content_ratings['5+'] = 4455
# MAGIC ````

# COMMAND ----------

content_ratings = {}
content_ratings['5+'] = 4455

print(content_ratings)

# COMMAND ----------

#To keep adding more values, we can do like this:

content_ratings = {}
content_ratings['4+'] = 4433
content_ratings['9+'] = 987
content_ratings['12'] = 1155
content_ratings['7+'] = 622

print(content_ratings)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Key-Value Pairs
# MAGIC 
# MAGIC A <b>key </b>is a index of a dictionary value. Such as in our example `'4+': 4433`, the dictionary key is `'4+'`, and the dictionary value is `4433`. As a whole, `'4+': 4433` is a <b>key-value pair</b>.
# MAGIC 
# MAGIC All of the data types such as strings, integers, floats, Booleans, lists, and even dictionaries can be dictionary values.

# COMMAND ----------

d_1 = { 'key_1' :'value_1',
        'key_2' :1,
        'key_3' :1.832,
        'key_4' :False,
        'key_5' :[1,2,3],
        'key_6' :{'inside key': 100}
}

print(d_1)
print(d_1['key_1'])
print(d_1['key_6'])

# COMMAND ----------

# MAGIC %md
# MAGIC Different than the dictionary values, dictionary keys cannot be [mutable types](https://towardsdatascience.com/https-towardsdatascience-com-python-basics-mutable-vs-immutable-objects-829a0cb1530a) like lists and dictionaries. A TypeError will be thrown if we try to use lists or dictionaries as dictionary keys.

# COMMAND ----------

d_1 = { 5 :'int',
        '5' :'string',
        3.5 :'float',
        False :'Boolean'}
print(d_1)

# COMMAND ----------

# A TypeError will be thrown
d_2 = {[1,2,3]:'list'}

# COMMAND ----------

# A TypeError will be thrown
d_2 = {{'key':'value'}:'dictionary'}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.5.4:
# MAGIC 
# MAGIC Create the following dictionary and assign it to a variable named ``d_1``:
# MAGIC 
# MAGIC {'key_1': 'first_value', <br>
# MAGIC  'key_2': 2, <br>
# MAGIC  'key_3': 3.14, <br>
# MAGIC  'key_4': True, <br>
# MAGIC  'key_5': [4,2,1], <br>
# MAGIC  'key_6': {'inner_key' : 6} <br>
# MAGIC  } <br>

# COMMAND ----------

### Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Counting with Dictionaries
# MAGIC 
# MAGIC The code below will show you how to update or change the dictionary values within an existing dictionary.

# COMMAND ----------

content_ratings = {'4+':4433, '9+':987, '12+':1155, '17+':622}

#Change the value corresponding to the dictionary key '4+' from 4433 to 1.
content_ratings['4+'] = 1

#Add 13 to the value corresponding to the dictionary key '9+'.
content_ratings['9+'] +=13

#Subtract 1128 from the value corresponding to the dictionary key '12+'.
content_ratings['12+'] -= 1128

#Change the value corresponding to the dictionary key '17+' from 622 (integer) to '511' (string).
content_ratings['17+'] = '511'

print(content_ratings)

# COMMAND ----------

# MAGIC %md
# MAGIC **Now we can combine the updating dictionary values technique with what we already know to count how many times each unique content rating occurs in our data set.** Let's start with this list: `['4+', '4+', '4+', '9+', '9+', '12+', '17+']`, which stores a few content ratings. To count how many times each rating occurs in this short list, we should:
# MAGIC - Create a dictionary where all the values (initial count) are all 0:`{'4+': 0, '9+': 0, '12+': 0, '17+': 0}`.
# MAGIC - Loop through the list `['4+', '4+', '4+', '9+', '9+', '12+', '17+']`, and for each iteration:
# MAGIC     - Check whether the iteration variable exists as a key in the previously created dictionary.
# MAGIC     - If it exists, then increment the dictionary value at that key by 1.

# COMMAND ----------

content_ratings = {'4+':0, '9+':0, '12+':0, '17+':0}
ratings = ['4+','4+','4+','9+','9+','12+','17+']

for c_rating in ratings:
    if c_rating in content_ratings:
        content_ratings[c_rating] += 1
content_ratings

# COMMAND ----------

# MAGIC %md
# MAGIC To get a better overview, we can print the `content_rating dictionary` inside the for loop and see the changes that it makes every time:

# COMMAND ----------

content_ratings = {'4+':0, '9+':0, '12+':0, '17+':0}
ratings = ['4+','4+','4+','9+','9+','12+','17+']

for c_rating in ratings:
    if c_rating in content_ratings:
        content_ratings[c_rating] += 1
    print(content_ratings)
print('Final dictionary:', content_ratings)

# COMMAND ----------

# MAGIC %md
# MAGIC How it's time to read in our data set (AppleStore.csv) and use the technique we learned above to count number of times each unique content rating occurs.
# MAGIC 
# MAGIC ### Task 1.5.5:
# MAGIC Count the number of times each unique content rating occurs in the data set.
# MAGIC 1. Create a dictionary named `content_ratings` where the keys are the unique content ratings and the values are all 0 (the values of 0 are temporary at this point, and they'll be updated).
# MAGIC 2. Loop through the `apps_data` list of lists. Make sure you don't include the header row. For each iteration of the loop:
# MAGIC     - Assign the content rating value to a variable named `c_rating`. The content rating is at index number 10 in each row.
# MAGIC     - Check whether ``c_rating`` exists as a key in ``content_ratings``. If it exists, then increment the dictionary value at that key by 1 (the key is equivalent to the value stored in `c_rating`).
# MAGIC 3. Outside the loop, print `content_ratings` to check whether the counting worked as expected.

# COMMAND ----------

opened_file = open('AppleStore.csv', encoding='utf8')
from csv import reader
read_file = reader(opened_file)
apps_data = list(read_file)

### Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Finding the Unique Values
# MAGIC 
# MAGIC In the example that we worked on in the previous session, we knew each of the unique values we wanted to count beforehand. This is not always the case. For example, what should we do if we don't have enough information to create the dictionary `{'4+': 0, '9+': 0, '12+': 0, '17+': 0}`?
# MAGIC 
# MAGIC We can update the code that we wrote for the list `['4+', '4+', '4+', '9+', '9+', '12+', '17+']` to accomodate this charateristic. 
# MAGIC In in addition to the `if` statement, we also added an `else` statement. If this dictionary key does not exist in our dictionary, then we create a new key-value pair in the `content_ratings` dictionary, where the dictionary key is the iteration variable (`c_rating`) and the dictionary value is 1.

# COMMAND ----------

content_ratings = {}
ratings = ['4+','4+','4+','9+','9+','12+','17+']

for c_rating in ratings:
    if c_rating in content_ratings:
        content_ratings[c_rating] += 1
    else:
        content_ratings[c_rating] = 1
content_ratings

# COMMAND ----------

# MAGIC %md
# MAGIC To get a better overview, we can print the `content_rating` dictionary inside the for loop and see the changes that it makes every time:

# COMMAND ----------

content_ratings = {}
ratings = ['4+','4+','4+','9+','9+','12+','17+']

for c_rating in ratings:
    if c_rating in content_ratings:
        content_ratings[c_rating] += 1
    else:
        content_ratings[c_rating] = 1
    print(content_ratings)
    
print('Final dictionary:')
content_ratings

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.5.6:
# MAGIC Now let's apply what we have just learned in our data set:
# MAGIC 
# MAGIC Count the number of times each unique content rating occurs in the data set while finding the unique values automatically.
# MAGIC 1. Create an empty dictionary named `content_ratings`.
# MAGIC 2. Loop through the `apps_data` list of lists (make sure you don't include the header row). For each iteration of the loop:
# MAGIC     - Assign the content rating value to a variable named `c_rating`. The content rating is at index number 10.
# MAGIC     - Check whether `c_rating` exists as a key in `content_ratings`.
# MAGIC         - If it exists, then increment the dictionary value at that key by 1 (the key is equivalent to the value stored in `c_rating`).
# MAGIC         - Else, create a new key-value pair in the dictionary, where the dictionary key is `c_rating` and the dictionary value is 3. Outside the loop, print `content_ratings` to check whether the counting worked as expected.

# COMMAND ----------

opened_file = open('AppleStore.csv', encoding='utf8')
from csv import reader
read_file = reader(opened_file)
apps_data = list(read_file)

### Start your code here:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Proportions and Percentages
# MAGIC 
# MAGIC The following table is known as the frequency table. A frequency is the number of times a unique value occurs.
# MAGIC 
# MAGIC |Content rating|Number of apps (frequency)|
# MAGIC |--|--|
# MAGIC |4+|4,433|
# MAGIC |9+|987|
# MAGIC |12+|1,155|
# MAGIC |17+|622|
# MAGIC 
# MAGIC 4+ occurs 4,433 times, so it has a frequency of 4,433. 12+ has a frequency of 1,155. 9+ has a frequency of 987. 17+ has the lowest frequency: 622.
# MAGIC 
# MAGIC When we analyze the data set, it might also be interesting to look at the following questions:
# MAGIC - What proportion of apps has a content rating of 4+?
# MAGIC - What percentage of apps has a content rating of 17+?
# MAGIC - What percentage of apps has a 15-year-old download?
# MAGIC 
# MAGIC To get the proportion of apps with a content rating of 4+, we can use the number of 4+ apps divide by the total number of apps like this:
# MAGIC 4,443/7,197
# MAGIC 
# MAGIC Instead of a fraction, we can also express proportion as a decimal between 0 and 1. So the result of 4,443/7,197 will be 0.62.
# MAGIC 
# MAGIC We can get percentage of 4+ apps by simply multiplying the proportions by 100 -- so it will be 62%.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Looping over Dictionaries
# MAGIC 
# MAGIC We can transform the frequencies to proportions or percentages individually by performing the required arithmetical operations like this:

# COMMAND ----------

content_ratings = {'4+':4433, '9+':987, '12+':1155, '17+':622}
total_number_of_apps = 7197

content_ratings['4+']/=total_number_of_apps
content_ratings['9+']/=total_number_of_apps
content_ratings['12+']/=total_number_of_apps
content_ratings['17+']/=total_number_of_apps

print(content_ratings)

# COMMAND ----------

# MAGIC %md
# MAGIC It can be very cumbersome to update each and every `content_rating` dictionary value manually. Therefore, we can use a for loop to iterate over a dictionary.

# COMMAND ----------

content_ratings = {'4+':4433, '9+':987, '12+':1155, '17+':622}

for iteration_variable in content_ratings:
    print(iteration_variable)

# COMMAND ----------

# MAGIC %md
# MAGIC We can also use the dictionary keys to access the dictionary values within the loop:

# COMMAND ----------

content_ratings = {'4+':4433, '9+':987, '12+':1155, '17+':622}

for iteration_variable in content_ratings:
    print(iteration_variable)
    print(content_ratings[iteration_variable])

# COMMAND ----------

# MAGIC %md
# MAGIC With the technique above, we can update the dictionary values within loop:

# COMMAND ----------

content_ratings = {'4+':4433, '9+':987, '12+':1155, '17+':622}
total_number_of_apps = 7197

for iteration_variable in content_ratings:
    content_ratings[iteration_variable] /= total_number_of_apps
print(content_ratings)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Small Bonus (OPTIONAL)
# MAGIC The looping through dictionaries which we used in this notebook is really used only for training purposes. In a productive and professional code, you would not want to use it. 
# MAGIC 
# MAGIC Here is a recommendation from our RBCZ colleague *Jakub Korecek*:
# MAGIC >  The dictionary type has one important and very useful method to check whether a key *exists*. This method is called ``get()``.
# MAGIC The useful part about this is that we can assign value if a key does not exist instead of having a KeyError. The default value is ``None``, but we can set up anything we want.

# COMMAND ----------

test_dict = {'A':1,'B':2}
print(test_dict.get('A'))
print(test_dict.get('C'))
print(test_dict.get('C','KEY DO NOT EXISTS'))

# COMMAND ----------

# MAGIC %md
# MAGIC > There is another way to iterate over a dictionary, which is better from performance point of view. It is done by method called ``items()``.

# COMMAND ----------

test_dict = {'A':1,'B':2}
for key,value in test_dict.items():
    print(key,value)
