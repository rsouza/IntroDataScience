# Databricks notebook source
# MAGIC %md
# MAGIC # Python Debugging Notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.Explanation 
# MAGIC This Jupyter notebook contains 10 tasks and is meant to be a checkpoint for your Python skill (on a very basic level). 
# MAGIC Each task is worth 1 point. You must achieve at least 7 points in order to pass this checkpoint. 
# MAGIC You may find helpful inserting a markdown cell if you feel like explaining your code or your thought process. 
# MAGIC
# MAGIC The notebook is inside of a folder called 05_Assessment. Inside the folder there are three files:
# MAGIC - Python_Debugging_Notebook.ipynb, which is the file you are currently looking at
# MAGIC - TestExample.png file that contains an useful example on how the feedback from the tests should be interpreted
# MAGIC - 2015.csv, where data for some of the exrcises is contained 
# MAGIC - tests.ipynb, which contains tests for checking your code - you can run these tests using the **last code cell at the bottom of this notebook**, you have to pass at least 7/10 tests
# MAGIC
# MAGIC ### 2.Your Task
# MAGIC - Fill out the **TODOs** tasks in each section of the Jupyter notebook. 
# MAGIC - Run all the cells and test your code at the bottom of this notebook - you should pass at least 7/10 tests
# MAGIC - Finally after you pass the tests, go back to the "Your turn with Git" section and push your branch as described there. 
# MAGIC
# MAGIC
# MAGIC Although most of the tasks should be easily solveable (considering you went through the previous notebooks) note that this is a **debugging notebook** and it is meant to make you solve and understand any errors you are getting (Google is your best friend). 
# MAGIC In case you have difficulties filling the TODOs, contact <renato.rocha-souza@rbinternational.com>.

# COMMAND ----------

# Importing paths
# Task 1:
import sys
import pandas as pd 
import pathlib
import os
from pathlib import Path


### TODO:
# - Locate/Find the 'potus.csv' file in the master folder structure and read it with the help of the read_csv() function 
#   which is loaded from the pandas library (example: pd.read_csv('my_folder/...some_path.../potus.csv'))
# - Store the result in a variable named 'df_potus'

# - Locate the 'f500.csv' file in the master folder structure and read it with the help of the pandas read_csv() function
# - Store the result in a variable named 'df_f500'
###

# Write your solution below:
#_____________________________________
potus_path = '../../../Data/potus.csv'
f500_path  = '../../../Data/f500.csv'

df_potus = pd.read_csv(potus_path)
df_f500 = pd.read_csv(f500_path)
df_f500.head()
# ____________________________________

# COMMAND ----------

# Installing libraries
# Task 2:

### TODO:
# - Not all Python libraries come pre-installed in Databricks like numpy and pandas, thus your task is 
#   to install a library called 'xgboost' using the command '!pip install' (example: !pip install superpowers)  
# - Import the newly installed 'xgboost' library using the 'import' keyword
# - Call the 'XGBClassifier()' function from the xgboost library using the dot notation (example: pandas.read_csv())
# - Store the called function in a variable named 'xgb'
###

# Write your solution below:
#_____________________________________
!pip install xgboost
import xgboost
xgb = xgboost.XGBClassifier()
# ____________________________________

# COMMAND ----------

# Writing Functions
# Task 3:

### TODO:
# - Write a function named 'multiply_by_three' that takes any positive integer (incl. the zero) as a parameter 
#   and returns that integer multiplied by 3 (example: 5 ==> 15); for any negative integer the function 
#   must return 0 (example: -5 ==> 0).
###

# Write your def function below:
#_____________________________________
def multiply_by_three(x):
    if( x < 0): 
        return x * 0
    return x * 3

# ____________________________________

print(multiply_by_three(1))

# COMMAND ----------

# Reading documentation and fixing bugs
# Task 4:
# Note: Task 1 cell must be run in order to proceed further with this task

### TODO:
# - Run the cell
# - Read about the drop function https://pandas.pydata.org/pandas-docs/version/0.21.1/generated/pandas.Panel.drop.html 
# - Fix the bug and remove the Country column from the list of columns (i.e. solve the KeyErorr) saving the result in 'df_2015_new'
###

df_2015 = pd.read_csv('../../../Data/2015.csv')                                        # Reading the data
print(f"Column values:\n[{(', ').join(df_2015.columns)}]", end='\n\n')   # Listing the columns
      
# Fix the buggy line of code below:
#_____________________________________
df_2015_new = df_2015.drop("Country", axis=1)                            # <--- Buggy line to remove the 'Country' column
# ____________________________________      
      
print(f"New Column values:\n[{(', ').join(df_2015_new.columns)}]")       # Listing the columns after the 'Country' column is removed


# Useful tip: You can use Shift + Tab on a function to see the inline documentation

# COMMAND ----------

# Extracting rows from a pandas dataframe
# Task 5:
# Note: Task 1 must be completed in order to proceed further with this task


### TODO:
# - Check the content of the first 5 rows of the alreaedy defined in task 1 'df_f500' datafranme
# - Save the 'Walmart' row to a variable named 'raw_walmart'
###

# Fix the buggy line of code below:
#_____________________________________
row_walmart = df_f500[df_f500['company']=="Walmart"]    
# ____________________________________

print(row_walmart)


# COMMAND ----------

# Traversal of data structures + conditional statements 
# Task 6:
# Note: You may find useful using for loops and if conditions


### TODO:
# - For every element in variable a, b below, if the element is equal to 'Python', consider it as a 'Yes'
# - How many 'Yesses' do you have? Store the number in a variale named 'cheeky'
### 

a = {'Python', 'R', 'SQL','Python', 'Git', 'Tableau', 'SAS', 'Python'}


b = ['Python', 'R', 'SQL', 'Python', 'Git', 'Tableau', 'SAS', 'Python', 'Python']

cheeky = 0
for x in a:
    if x == 'Python':
        cheeky +=1
        print("list a")
    
for x in b:
    if x == 'Python':
        cheeky +=1
        print("list b")
# Don't forget to define the variable 'cheeky' # too many tips?

print(cheeky)

# COMMAND ----------

# Tasks 7 & 8
# Hint: Be careful about the data types(integers, strings, etc.) when instantiating the objects

### TODO:
# Task 7:
#  - Instantiate a bank account object with the following characteristics:
#       - account name - 'RBI', currency - 'Euro' and balance 23 000
#       - store the object in a variable named 'bank_account_1'
#  - Use the deposit() method to deposit 2 000 into the bank account
#  - Print the balance of the account

#  - Instantiate a bank account object with the following characteristics:
#       - account name - 'BT', currency - 'Euro' and balance 900
#       - store the object in a variable named 'bank_account_2'
#  - Use the withdraw() method to withdraw 400 from the bank account
#  - Print the balance of the account

# Task 8:
#  - Calculate the total balance of 'bank_account_1' and 'bank_account_2'
#  - Store the results in a variable named 'total_balance'
###

class BankAccount:
    def __init__(self, account_name, currency, balance=0):
        self.account_name = account_name
        self.currency = currency
        self.balance = balance
    
    def rename_account_name(self, value):
        self.balance = value
        
    def deposit(self, value):
        self.balance = self.balance + value

    def withdraw(self, value):
        self.balance = self.balance - value

    def get_balance(self):
        return self.balance

    
# Your code goes below:
#_____________________________________

def bank_add(a,b):
    return a.get_balance() + b.get_balance()
bank_account_1 = BankAccount('RBI', 'Euro', balance = 23000)
bank_account_1.deposit(2000)
print(bank_account_1.get_balance())

bank_account_2 = BankAccount('BT', 'Euro', balance = 900)
bank_account_2.withdraw(400)
print(bank_account_2.get_balance())

#total_balance = bank_account_1.get_balance() + bank_account_2.get_balance()
total_balance = bank_add(bank_account_1, bank_account_2)
print(total_balance)
# ____________________________________


# COMMAND ----------

# Dealing with data types (strings, integers, etc.)
# Task 9:
# Sometimes you may find yourself having to convert data types, this is also known as 'casting'. Always check your data types.
# People love to store numeric and datetime values as text :( 


### TODO:
# - Run the cell
# - Find the value of the algebraic difference between the largest happiness rank and the smallest happiness rank 
#   in the '2015.csv' dataset. You may find useful using the max() and min() built-in functions.
# - Store the result in a variable named 'diff'
## 

df_2015 = pd.read_csv('../../../Data/2015.csv') 
df_2015['Happiness Rank'] = df_2015['Happiness Rank'].astype('str')
df_2015.head()                         # Note that this loads only the first 5 rows of the dataframe

# Your code goes below:
#_____________________________________
m = max(df_2015['Happiness Rank'])
s = min(df_2015['Happiness Rank'])
print(m)
print(s)


# diff = ord(str(df_2015[df_2015['Happiness Rank'] == m]['Country'])) - ord(str(df_2015[df_2015['Happiness Rank'] == s]['Country']))
diff = df_2015['Happiness Rank'].astype(int).max() - df_2015['Happiness Rank'].astype(int).min()

# ____________________________________

print(diff)

type(diff)

# COMMAND ----------

# Google expoloration on how to solve a task

# Task 10:

### TODO:
# - Write some short lines of Python code below to print out the current date. 
#       Every time when the 'run' button is hit on this cell, the most up-to-date date will be generated 
#       without you manually updating the code. 
# - Subsract 5 days from the current date and store the result in a variable called 'five_days_ago'. 
#       Note that manually writing the date (example: five_days_ago="2021-01-01") will result in a failed test. 
#       You have to dynamically substract 5 days every time you run the cell. 
#       You may want to use the timedelta function to do this.
###


#Subtract 5 days from the current date : ----> not too clear, what does it mean

# Your code goes below:
#_____________________________________
import datetime
print(datetime.datetime.now().date())
from datetime import date, timedelta
five_days_ago = date.today() - timedelta(5)

# ____________________________________

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.Tests
# MAGIC Please run the cell below (the last cell of this notebook) to test your solutions. 
# MAGIC Note that you will have to **run all previous cells** in order to load the varaibles that will be checked. 
# MAGIC
# MAGIC You will see a message next to each task - "ok" means you pass the test. 
# MAGIC You have to pass at least 7 out of the 10 tests. Good luck! 

# COMMAND ----------

# MAGIC %run ./tests
