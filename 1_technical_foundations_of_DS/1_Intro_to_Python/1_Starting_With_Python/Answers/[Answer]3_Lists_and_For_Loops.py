# Databricks notebook source
# MAGIC %md
# MAGIC ## About This Notebook
# MAGIC 
# MAGIC In the previous notebook we learned what variables are and how these can store various values (strings, integers ...). 
# MAGIC 
# MAGIC - In this notebook, we will expand our toolset with a more complex type of variable - **list**. As you will see, a list can store many elements (not just one like a variables in the previous lecture). 
# MAGIC - We will learn later on how to manipulate these lists, such as how to navigate through them and how to retrieve elements from them. 
# MAGIC - In the final part of the notebook we will hop on to a different concept - **loops**. These are closely connected to the topic of lists (and for example retrieval from lists). If the list is *very* long it might become unfeasible to retrieve elements from it manually. We will need loops for this.
# MAGIC 
# MAGIC Please note that in this notebook you will see **libraries being imported**. To understand what these imports of libraries are (with all the functions and commands) is not yet required knowledge. Try to follow only the examples which are shown.
# MAGIC ***

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Lists
# MAGIC 
# MAGIC From last chapter we have worked with the table below:
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
# MAGIC There are 5 rows and 5 columns in this table. Each value in this table is called a **data point**. For example in the first row, we have five data points and they are:
# MAGIC 
# MAGIC - Facebook
# MAGIC - 0.0
# MAGIC - USA
# MAGIC - 2974676
# MAGIC - 3.5

# COMMAND ----------

# MAGIC %md
# MAGIC When we have a collection of data points it is called a **data set**. We can understand the table above as a collection of data points, so we can call the entire table a data set as well. We see that the data set has five rows and five columns. 
# MAGIC 
# MAGIC When we want to work with data sets in the computer, we need to store them properly in the computer memory so we can retrieve and manipulate the data points according to our needs. To store each data point in the computer, we can for instance take what we have learned so far and do it this way:

# COMMAND ----------

track_name_row1 = 'Facebook'
price_row1 = 0.0
currrency_row1 = 'USD'
rating_count_total_row1 = 2974676
user_rating_row1 = 3.5

# COMMAND ----------

# MAGIC %md
# MAGIC **To enter each data point manually in the data set like we did above would involve a lot of labor and is nearly impossible for a big data set. Fortunately, we can store data in another more efficient way by using lists.** See example below:

# COMMAND ----------

row_one = ['Facebook', 0.0, 'USD', 2974676, 3.5]
print(row_one)
type(row_one)

# COMMAND ----------

# MAGIC %md
# MAGIC How did we create the list above? 
# MAGIC 
# MAGIC 1. Enter a sequence of data points that we want to include in our data set and **separated each with a comma**: 'Facebook', 0.0, 'USD', 2974676, 3.5
# MAGIC 2. Use **brackets for surrounding** the sequence of data points
# MAGIC 
# MAGIC After we created the list, the list is stored in the computer's memory by assignment to the variable name row_one. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.3.1
# MAGIC 1. Store the second row as a list in a variable named `row_two`.
# MAGIC 2. Store the third row as a list in a variable named `row_three`.
# MAGIC 3. Print out `row_two` and `row_three`.

# COMMAND ----------

# MAGIC %md
# MAGIC | Track_name |  Price |  Currency |  Rating_count_total | User_rating|
# MAGIC |------------|:------:|----------:|---------------------:|-----------:|
# MAGIC | Facebook | 0.0 | USD | 2974676 | 3.5|
# MAGIC | Instagram |    0.0  |   USD |2161558 |4.5|
# MAGIC | Clash of Clans | 0.0|    USD | 2130805 |4.5|
# MAGIC | Temple Run |    0.0  |   USD |1724546 |4.5|
# MAGIC | Pandora - Music & Radio | 0.0|    USD | 1126879 |4.5|

# COMMAND ----------

# Start your code below:

row_two = ["Instagram", 0.0, "USD", 2161558, 4.5]
row_three = ["Clash of Clans", 0.0, "USD", 2130805, 4.5]
print(row_two, row_three)

# COMMAND ----------

# MAGIC %md
# MAGIC A list can contain both -- mixed and identical data types.
# MAGIC 
# MAGIC For example, a list that contains only integers looks like this: `[ 1, 2, 3]`. A mixed data type list looks like the one we created above: `['Facebbok', 0.0, 'USD', 2974676, 3.5]`. In this list, we have:
# MAGIC - Two strings ('Facebook', 'USD')
# MAGIC - Two floats (0.0, 3.5)
# MAGIC - One integer (2974676)
# MAGIC 
# MAGIC In order to find out the **length of a list, we can use the `len()` command**. 
# MAGIC The row_one list for example has five data points.

# COMMAND ----------

row_one = ['Facebook', 0.0, 'USD', 2974676, 3.5]
print(len(row_one))

list_a = [1,2,3]
print(len(list_a))

list_b = []
print(len(list_b))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Indexing a List
# MAGIC 
# MAGIC **Each data point or element in the list has a specific index number (position) associated with it**. In Python, the indexing always starts with 0. For example, the first element always has the index number 0, and the second element always has the index number 1, and so on.
# MAGIC 
# MAGIC To find the index of an element in the list, simply count the position of the index and then minus one. For example, the user rating is the 5th element in the list and it has an index number of 4 (5 - 1 = 4).
# MAGIC 
# MAGIC *Note: Please really write this down - the indexing in Python starts at 0.*

# COMMAND ----------

row_one = ['Facebook', 0.0, 'USD', 2974676, 3.5]
print(row_one[4])

# COMMAND ----------

# MAGIC %md
# MAGIC The index number can help us retrieve individual elements from a list. So how do we retrieve the first element? We can enter the list name and then wrap the index number in a bracket. The general formula follows the model:  `list_name[index_number]`. To retrieve each element from the list is useful when we want to perform operations between the elements. For example, we can find out the average and difference between Facebook's and Instagram's ratings by doing this: 

# COMMAND ----------

row_one = ['Facebook', 0.0, 'USD', 2974676, 3.5]
row_two = ['Instagram',0.0, 'USD', 2161558, 4.5]

difference_in_rating = row_two[4] - row_one[4]
average_rating = (row_one[4] + row_two[4])/2

print(difference_in_rating)
print(average_rating)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.3.2
# MAGIC 
# MAGIC In the code editor below, you can see the lists for the first three rows from the table. Retrieve the fourth element from each list, which represents the rating each app has received, and fnid the average value of the retrieved numbers.
# MAGIC 
# MAGIC 1. Assign the fourth element from the list ``row_one`` to a variable named ``rating_one``. Don't forget that the indexing starts at 0. <br>
# MAGIC 2. Repeat the same step for the fourth element from the list ``row_two``, to a variable named ``rating_two``.<br>
# MAGIC 3. Repeat the same step for for the fourth element from the list ``row_three``, to a variable named  ``rating_three``.<br>
# MAGIC 4. Add the three numbers from above steps together and save the value to a variable named ``total_rating``.
# MAGIC 5. Divide the ``total_rating`` by 3 to get the average number of ratings for the first three rows. Assign the final result to a variable named ``average_rating``.
# MAGIC 6. Print `average_rating`.

# COMMAND ----------

row_one = ['Facebook', 0.0, 'USD', 2974676, 3.5]
row_two = ['Instagram', 0.0, 'USD', 2161558, 4.5]
row_three = ['Clash of Clans', 0.0, 'USD', 2130805, 4.5]

# Start your code below:
rating_one = row_one[3]
rating_two = row_two[3]
rating_three = row_three[3]
total_rating = rating_one + rating_two + rating_three
average_rating =  total_rating / 3
print(average_rating)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Negative Indexing
# MAGIC 
# MAGIC There are two indexing systems for lists in Python.
# MAGIC - Positive indexing: the one we have encountered so far starting with the first element having the index number 0 and the second element having the index number 1, and so on.
# MAGIC - Negative indexing: starting with the **last element** having the index number -1, the second to last element having the index number -2, and so on.
# MAGIC 
# MAGIC Positive indexing is often put into practice in day to day programming. However, negative indexing can also come in handy when we want to select the last element of a list, especially when this list is long and we have no idea how many elements it contains.

# COMMAND ----------

row_one = ['Facebook', 0.0, 'USD', 2974676, 3.5]

print(row_one[-1])
print(row_one[4])

# COMMAND ----------

# MAGIC %md
# MAGIC See that the code above, indexing number -1 and 4 retrieve the same result. 
# MAGIC Pay attention: if we try to use an index number that is outside the range of the two indexing systems that are mentioned above, we'll get the following error:

# COMMAND ----------

print(row_one[5])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.3.3
# MAGIC 
# MAGIC The last element in each list represents the rating of each app.
# MAGIC 
# MAGIC 1. Retrieve the ratings for the first three rows (like what you have done in task 2), and then find the average of all the ratings retrieved and print it to the screen.
# MAGIC 
# MAGIC <b>Try to use the negative indexing system.</b>

# COMMAND ----------

row_1 = ['Facebook', 0.0, 'USD', 2974676, 3.5]
row_2 = ['Instagram', 0.0, 'USD', 2161558, 4.5]
row_3 = ['Clash of Clans', 0.0, 'USD', 2130805, 4.5]

##Your task starts here:
rating_one = row_1[-1]
rating_two = row_2[-1]
rating_three = row_3[-1]

average_rating = (rating_one + rating_two + rating_three)/3
print(average_rating)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Retrieving Multiple List Elements
# MAGIC There are occasions where we are asked to retrieve more than just one element from a list. 
# MAGIC Take the list we have already worked with before: `['Facebook', 0.0, 'USD', 2974676, 3.5]`. We're  interested in knowing only the name of the app and the data about ratings (the number of ratings and the rating). Taking what we have learned so far, we can do this as follows:

# COMMAND ----------

row_one = ['Facebook', 0.0, 'USD', 2974676, 3.5]
app_name = row_one[0]
rating_count_total = row_one[3]
rating = row_one[-1] # or rating = row_one[4] 

# COMMAND ----------

# MAGIC %md
# MAGIC It will be very inefficient if we try to do this for every app in the data table and make our code unnecessarily lengthy and hard to keep track of. 
# MAGIC Alternatively we can store all the data we want to keep track of in a separate list like this:

# COMMAND ----------

row_one = ['Facebook', 0.0, 'USD', 2974676, 3.5]

fb_rating = [row_one[0], row_one[3], row_one[-1]]
print(fb_rating)

# COMMAND ----------

# MAGIC %md
# MAGIC In the code above we have successfully abstracted all the information we needed from the list into a separate list.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.3.4
# MAGIC 
# MAGIC 1. Abstract the rating data for Facebook, Instagram, Pandora - Music & Radio into separate lists. Each list should contain information like: the name of the app, the total rating count, and the user rating. Don't forget the positive index system begins with indexing number 0.
# MAGIC   - For Facebook, assign the list to a variable called `fb_rating_data`
# MAGIC   - For Instagram, assign the list to a variable called `insta_rating_data`
# MAGIC   - For Pandora - Music & Radio, assign the list to a variable called `pan_rating_data`
# MAGIC 
# MAGIC 
# MAGIC 2. Compute the average user rating for Instagram, and Pandora — Music & Radio using the data you just created ``fb_rating_data``, ``insta_rating_data``, and ``pan_rating_data``.
# MAGIC   - Sum up all the ratings together and compute the average rating.
# MAGIC   - Assign the result to a variable named `avg_rating`.

# COMMAND ----------

row_one = ['Facebook', 0.0, 'USD', 2974676, 3.5]
row_two = ['Instagram', 0.0, 'USD', 2161558, 4.5]
row_three = ['Clash of Clans', 0.0, 'USD', 2130805, 4.5]
row_four = ['Temple Run', 0.0, 'USD', 1724546, 4.5]
row_five = ['Pandora - Music & Radio', 0.0, 'USD', 1126879, 4.0]

# Start your code below:

# Question 1:
fb_rating_data = [row_one[0], row_one[3], row_one[-1]]
insta_rating_data = [row_two[0], row_two[3], row_two[-1]]
pan_rating_data = [row_five[0], row_five[3], row_five[-1]]

# Question 2:
avg_rating = (fb_rating_data[-1] + insta_rating_data[-1] + pan_rating_data [-1])/3
print(avg_rating)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. List Slicing (IMPORTANT)
# MAGIC 
# MAGIC In the previous exercise, we retrieved the first, fourth, and the last element and put them into a separate list.
# MAGIC We can also retrieve the first three elements from the list and create a separate pricing data list.
# MAGIC 
# MAGIC For example:

# COMMAND ----------

row_three = ['Clash of Clans', 0.0, 'USD', 2130805, 4.5]

clash_pricing_data = [row_three[0], row_three[1], row_three[2]]
print(clash_pricing_data)

# COMMAND ----------

# MAGIC %md
# MAGIC However, there is a shortcut in selecting the first three elements like this:

# COMMAND ----------

row_three = ['Clash of Clans', 0.0, 'USD', 2130805, 4.5]

clash_pricing_data = row_three[0:3] #This is also known as the syntax shortcut
print(clash_pricing_data)

# COMMAND ----------

# MAGIC %md
# MAGIC As shown in the code, if we want to select the first <b> n</b> elements from a list called list_one, we can use the syntax shortcut `list_one[0:n]`. 
# MAGIC In the example above, we have selected the first three elements from the list called `row_three`, so we have used: `row_3[0:3]` .
# MAGIC The process of selecting a particular part of a list is known as list slicing.
# MAGIC 
# MAGIC In order to retrieve any list slice we want, we need to:
# MAGIC 1. Identify the first and last element of the slice.
# MAGIC 2. Identify the index numbers of the first and last element of the slice.
# MAGIC 3. We can then retrieve the list slice we want by using the syntax we just mentioned above like: `list_one[m:n]`, where:
# MAGIC     - m is the index number of the first element of the slice
# MAGIC     - n is the index number of the last element of the slice <b> plus one </b>. (That means if the last element has the index number 5, then <b>n</b> will be 6, and if the last element has the index number 10, then<b> n</b> will be 11, and so on)

# COMMAND ----------

# MAGIC %md
# MAGIC There is an even simpler syntax shortcuts when we want to select the first or last n elements (n stands for an integer):
# MAGIC - `list_one[:n]`: when we want to select the first n elements
# MAGIC - `list_one[-n:]`: when we want to select the last n elements

# COMMAND ----------

row_three = ['Clash of Clans', 0.0, 'USD', 2130805, 4.5]

first_3 = row_three[:3] #select the first 3 elements
last_3 = row_three[-3:] #select the last 3 elements

print(first_3)
print(last_3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.3.5
# MAGIC 
# MAGIC 1. Select the first four elements from `row_one` using the list slicing syntax shortcut and assign the result to a variable named `first_four_elem_fb`.
# MAGIC 2. Select the last two elements from `row_one` using the list slicing syntax shortcut and assign the result to a variable named `last_two_elem_fb`.
# MAGIC 3. For `row_five`, select the list slice `[ 0.0, 'USD', 1126879]` using the list slicing syntax shortcut and assign the result to a variable named `pan_2_3_4`.

# COMMAND ----------

row_one = ['Facebook', 0.0, 'USD', 2974676, 3.5]
row_two = ['Instagram', 0.0, 'USD', 2161558, 4.5]
row_three = ['Clash of Clans', 0.0, 'USD', 2130805, 4.5]
row_four = ['Temple Run', 0.0, 'USD', 1724546, 4.5]
row_five = ['Pandora - Music & Radio', 0.0, 'USD', 1126879, 4.0]


# Start your code below:

first_four_elem_fb = row_one[:4]
last_two_elem_fb = row_one[-2:]
pan_2_3_4 = row_five[1:4]

# print(first_four_elem_fb)
# print(last_two_elem_fb)
# print(pan_2_3_4)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. List of Lists
# MAGIC So far we have only worked with a data set of small scale (only five rows!) and we have stored each row as a list in a seprate variable like `row_one`, `row_two`, `row_three`, `row_four`, `row_five`. However, if we had a data set with more than 10,000 rows, then we'd have ended up with 10,000 more variables which is unrealistic and redundant.
# MAGIC 
# MAGIC To solve this problem, we can store all five lists in one single variable like this:

# COMMAND ----------

row_one = ['Facebook', 0.0, 'USD', 2974676, 3.5]
row_two = ['Instagram', 0.0, 'USD', 2161558, 4.5]
row_three = ['Clash of Clans', 0.0, 'USD', 2130805, 4.5]
row_four = ['Temple Run', 0.0, 'USD', 1724546, 4.5]
row_five = ['Pandora - Music & Radio', 0.0, 'USD', 1126879, 4.0]

data_set = [row_one, row_two, row_three, row_four, row_five]
data_set

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that now the variable data_set contains all five lists `row_one, row_two, row_three, row_four, row_five`. Such a list that contains other lists is called a <b> list of lists </b>.
# MAGIC 
# MAGIC `data_set` is declared as a list, so we can retrieve the individual list elements and perform list slicing with what we have learned.

# COMMAND ----------

data_set = [row_one, row_two, row_three, row_four, row_five]

#To retrieve the first element
print(data_set[0])

#To retrieve the last element
print(data_set[-1])

#To retrieve the first two list elements 
print(data_set[:2])

# COMMAND ----------

# MAGIC %md
# MAGIC There are also cases where we need to retrieve individual elements from a list that's part of a list of lists — for instance, we may want to retrieve the value 'Facebook' from ['Facebook', 0.0, 'USD', 2974676, 3.5], which is now part of the `data_set` list of lists.

# COMMAND ----------

data_set = [row_one, row_two, row_three, row_four, row_five]

#We can retrieve row_one and assign it to a variable like this:
fb_list = data_set[0]

#To print out the facebook list data
print(fb_list)

#To retrieve the first element from fb_list
app_name = fb_list[0]

#Print the result of app_name
print(app_name)

# COMMAND ----------

# MAGIC %md
# MAGIC In the code above, we have retrieved the first element in two steps: we first retrieved `data_set[0]`, and then we retrieved `fb_row[0]`.
# MAGIC There is a much simpler way to retrieve the value 'Facebook' by chaining the two indices ([0] and [0])
# MAGIC For example `data_set[0][0]` will give you the same output.

# COMMAND ----------

data_set = [row_one, row_two, row_three, row_four, row_five]

print(data_set[0] [0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.3.6
# MAGIC 
# MAGIC 
# MAGIC 1. In the code editor below please group together the five lists into a **list of lists** named ``app_data_set``.
# MAGIC 2. Compute the average rating of the apps by retrieving the right data points from the ``app_data_set`` list of lists.
# MAGIC   - The user rating is always the last element of each row. Sum up the ratings and then divide by the number of ratings to get the average.
# MAGIC   - Assign the result to a variable named ``average_rating``.
# MAGIC   - Print `average_rating`.

# COMMAND ----------

row_1 = ['Facebook', 0.0, 'USD', 2974676, 3.5]
row_2 = ['Instagram', 0.0, 'USD', 2161558, 4.5]
row_3 = ['Clash of Clans', 0.0, 'USD', 2130805, 4.5]
row_4 = ['Temple Run', 0.0, 'USD', 1724546, 4.5]
row_5 = ['Pandora - Music & Radio', 0.0, 'USD', 1126879, 4.0]


# Start your code below:

app_data_set= [row_1, row_2, row_3, row_4, row_5]
print(app_data_set)
total_rating = app_data_set[0][-1] + app_data_set[1][-1]+ app_data_set[2][-1]+app_data_set[3][-1]+app_data_set[4][-1]
print(total_rating)
average_rating = total_rating/5
print(average_rating)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Opening a File (OPTIONAL)
# MAGIC 
# MAGIC The data set we've been working with so far is a small snippet from a much larger data set from Kaggle. See data source: [Mobile App Store Data Set (Ramanathan Perumal)](https://www.kaggle.com/ramamet4/app-store-apple-data-set-10k-apps). 
# MAGIC The data set contains 7,197 rows and 16 columns, which amounts to 115,152 (7,197 * 16) data points. It would be impossible to type all the data points manually into our computer and have them saved in the computer memory. However, we can always download a file and open it in our computer by using the `open()` command. For example if we have a file saved on the computer called "AppleStore.csv", we could open it like this:
# MAGIC 
# MAGIC *my_file = open('AppleStore.csv')*
# MAGIC 
# MAGIC Once we've opened the file we can read it in using a command called `reader()`. We need to import the `reader()` command from the **csv module** using the code `from csv import reader` (a **module is a collection of commands and variables**). Then we can transform the file into a list of lists using the `list()` command. The entire process looks like this:

# COMMAND ----------

my_file = open('AppleStore.csv', encoding='utf8')

from csv import reader
read_file = reader(my_file)
apps_data = list(read_file)

#To print out the first five rows of apps_data
print(apps_data[:5])

#To find out the length of our data set
len(apps_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.1. Dealing with files with the context manager
# MAGIC We started to open CSV files. We did so using a very simplified snippet of code, which would not be the case in professional code. Normally we work with the context manager `with`.  
# MAGIC The Python `with` statement creates a runtime context that allows you to run a group of statements under the control of a context manager. In case of opening files, the `with` closes them automatically for us at the end of the block.
# MAGIC 
# MAGIC Each file should be closed once we finished the work -- that will release resources back to OS. It is not necessary in case of few files (such as in these training notebooks), but the best practices is to do so everytime. It will prevent many possible problems in your future career as Data Scientist. We can use two following ways:

# COMMAND ----------

# context manager
with open('AppleStore.csv', encoding='utf8') as my_file: 
    from csv import reader
    read_file = reader(my_file)
    apps_data = list(read_file)

    #To print out the first five rows of apps_data
    print(apps_data[:5])

    #To find out the length of our data set
    len(apps_data)

    
# or a more classical approach      
print("*** Second way to do it ***")
my_file_2 = open('AppleStore.csv', encoding='utf8')
read_file = reader(my_file_2)
apps_data = list(read_file)
print(apps_data[:5])
len(apps_data)

# closing resources
my_file_2.close() 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Repetitive Processes
# MAGIC 
# MAGIC In the previous task, we have retrieved ratings manually. However, try to retrieve 7,197 ratings manually is impractical because it takes a lot of time. How can we find a way that retrieve all 7,197 ratings in just a couple seconds? Luckily, Python offers us an easy way to repeat a process. Let's look at the following example:

# COMMAND ----------

ratings = [3, 10, 2, 6, 8, 1]

for element in ratings:
    print(element)

# COMMAND ----------

# MAGIC %md
# MAGIC Do you remember the first example we had above? Where we wanted to retrieve the last element for each list in `app_data_set`. We can translate this process using the for loop in Python, like this:

# COMMAND ----------

app_data_set = [row_one, row_two, row_three, row_four, row_five]

for each_list in app_data_set:
    rating = each_list[-1]
    print(rating)

# COMMAND ----------

# MAGIC %md
# MAGIC The code above  takes each list element from `app_data_set` and assigns it to ``each_list`` (which basically becomes a variable that stores a list). See in the example below how each element in the ``app_data_set`` is taken out (or isolated) and gets printed onto the screen:

# COMMAND ----------

app_data_set = [row_one, row_two, row_three, row_four, row_five]

for each_list in app_data_set:
    print(each_list)

# COMMAND ----------

# MAGIC %md
# MAGIC To make it more understandable, it basically does the following:

# COMMAND ----------

app_data_set = [row_one, row_two, row_three, row_four, row_five]

print(app_data_set[0])
print(app_data_set[1])
print(app_data_set[2])
print(app_data_set[3])
print(app_data_set[4])

# COMMAND ----------

# MAGIC %md
# MAGIC The for loop technique requires us to write only two lines of code regardless of the number of rows in the data set, even if the data set has couple millions rows! :)
# MAGIC 
# MAGIC One thing to pay attention to: do not forget to **indent** the code after the `for` line. To indent means to use for example TAB at the beginning of next line if your editor does not do it automatically for you.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.3.8: 
# MAGIC Use the new technique we've learned to print all the rows in the `app_data_set` list of lists.

# COMMAND ----------

row_1 = ['Facebook', 0.0, 'USD', 2974676, 3.5]
row_2 = ['Instagram', 0.0, 'USD', 2161558, 4.5]
row_3 = ['Clash of Clans', 0.0, 'USD', 2130805, 4.5]
row_4 = ['Temple Run', 0.0, 'USD', 1724546, 4.5]
row_5 = ['Pandora - Music & Radio', 0.0, 'USD', 1126879, 4.0]

app_data_set = [row_1, row_2, row_3, row_4, row_5]

# Start your code below:

for x in app_data_set:
    print(x)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. For Loops
# MAGIC 
# MAGIC What we just learned above is called <b>for loop</b>, as we have already mentioned.
# MAGIC 
# MAGIC There are three structural parts in a <b>for loop</b>:
# MAGIC 1. iteration variable
# MAGIC 2. iterable variable
# MAGIC 3. the body of the for loop

# COMMAND ----------

a_list = [1, 2, 3]

for value in a_list:
    print(value)
    print(value - 1)

# value - is the iteration variable
# a_list - is the iterable variable
# print(value) & print(value - 1) - belong to the body of the loop

# COMMAND ----------

# MAGIC %md
# MAGIC The indented code in the <b>body</b> gets executed the same number of times as elements in the <b>iterable variable</b>. If the iterable variable is a list that has three elements, the indented code in the body gets executed three times. We call each code execution an<b> iteration</b>, so there'll be three iterations for a list that has three elements. For each iteration, the <b>iteration variable</b> will take a different value, following this pattern:
# MAGIC 
# MAGIC - In the first iteration, the value is the first element of the iterable (we have the list `[1, 2, 3]` as the iterable, so the value will be 1).
# MAGIC - In the second iteration, the value is the second element of the iterable (we have the list `[1, 2, 3]` as the iterable, so the value will be 2).
# MAGIC - In the third iteration, the value is the third element of the iterable (we have the list `[1, 2, 3]` as the iterable, so the value will be 3).
# MAGIC 
# MAGIC Do you know that the code outside the loop body can also interact with the code inside the loop body? Look at the following example:

# COMMAND ----------

a_list = [1, 2, 3]

a_sum = 0 # a variable outside the loop body

for value in a_list: # for every iteration of the loop
    ######################Everything below is inside the loop body###################
    
    a_sum = a_sum + value # perform an addition between the current value 
    # of the iteration variable and the current value stored in a_sum, 
    # and assign the result of the addition back to a_sum
    print(a_sum) # print the value of the a_sum variable
    
    ######################Everything above is inside the loop body###################

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.3.9:
# MAGIC Compute the average app rating for the apps stored in the `app_data_set` variable.
# MAGIC 
# MAGIC 1. Initialize a variable named `rating_sum` with a value of zero outside the loop body.
# MAGIC 2. Loop (iterate) over the `app_data_set` list of lists. For each of the five iterations of the loop (for each row in `app_data_set`):
# MAGIC 3. Extract the rating of the app and store it to a variable named `rating`. The rating is the last element of each row.
# MAGIC 4. Add the value stored in rating to the current value of the `rating_sum`.
# MAGIC 5. Outside the loop body, divide the rating sum (stored in `rating_sum`) by the number of ratings to get an average value. Store the result in a variable named `avg_rating`.

# COMMAND ----------

row_1 = ['Facebook', 0.0, 'USD', 2974676, 3.5]
row_2 = ['Instagram', 0.0, 'USD', 2161558, 4.5]
row_3 = ['Clash of Clans', 0.0, 'USD', 2130805, 4.5]
row_4 = ['Temple Run', 0.0, 'USD', 1724546, 4.5]
row_5 = ['Pandora - Music & Radio', 0.0, 'USD', 1126879, 4.0]

app_data_set = [row_1, row_2, row_3, row_4, row_5]

rating_sum = 0.

for row in app_data_set:
    rating = row[4]
    rating_sum +=rating
    
avg_rating = rating_sum / len(app_data_set)
