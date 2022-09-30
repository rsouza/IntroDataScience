# Databricks notebook source
# MAGIC %md
# MAGIC ## About This Notebook
# MAGIC 
# MAGIC Within this notebook, we are going to make our very first steps into the vast world of Python. 
# MAGIC 
# MAGIC - What does one line of *code* looks like?
# MAGIC - What is *syntax*?
# MAGIC - What are *comments*?
# MAGIC - How do we perform *arithmetic operations*?
# MAGIC ***

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. First Lines of Code

# COMMAND ----------

# MAGIC %md
# MAGIC Python is one of the most popular programming langugage these days. It designed for readability, and it shares some similarities with the English Language as well as having influence from mathematics. It is one of the most straight-forward and easily understandable programming languages.
# MAGIC 
# MAGIC 
# MAGIC ### Task 1.1.1:
# MAGIC 1. How does Python compute 25+5? Try type in 25 + 5 in the following cell, just below where the text indicates it:

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC 2. In the line above, a single calculation is instructed: 25 + 5. However, the computer is capable to perform more than just one calculation. Try to instruct the computer to perform multiple calculations like:
# MAGIC 
# MAGIC 25 + 5 <br>
# MAGIC 20 - 7 <br>
# MAGIC 30 + 2 <br>
# MAGIC 67 - 22 <br>

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC The output is 45 and it seems like the computer only performed the last subtraction, 67 - 22. Hmm strange...
# MAGIC ## 2. The print() Command
# MAGIC 
# MAGIC What really happens is that the computer performed all the calculations above, but it only displays the last one as the output result. To display all calculation results, what we need is to use the `print()` command (function), like this:

# COMMAND ----------

print(25 + 5)
print(10 - 6)
print(30 + 2)
print(12 + 38)

# COMMAND ----------

# MAGIC %md
# MAGIC Don't worry for now what *command* (or function) exactly is. You will get to learn that later. It is for you, for now, only important to understand that if we learn what a command does, we can use it to our advantage.Now let's have some practice with the `print()` command.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.1.2:
# MAGIC 1. Using the `print()` command and display the result for:
# MAGIC -  55 + 5
# MAGIC - 300 - 8
# MAGIC - 21 + 67
# MAGIC 
# MAGIC 2. Click the Run button when you're ready to see your results.

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC Have you ever wondered what happens if we put all print commands one the same line? 
# MAGIC Try to type this in the following cell: 
# MAGIC `print(55 + 5) print(300 - 8) print(21 + 67)`

# COMMAND ----------

print(55 + 5) print(300 - 8) print(21 + 67)

# COMMAND ----------

# MAGIC %md
# MAGIC It would work, however, if we had separated the commands with a semicolon (although this is not usual in Python).

# COMMAND ----------

print(55 + 5); print(300 - 8); print(21 + 67)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Python Syntax
# MAGIC Yes, indeed. We'd get an **error**. And in fact it is described as a syntax error. This is because all programming languages like Python, Java, C++ all have its own **syntax rules**. Each line of instructions must comply with these rules. 
# MAGIC 
# MAGIC You can compare these syntax rules with grammar in human languages. If we want to convey a message, we must respect and follow the syntax rules in order to deliver our message in a meaningful way. For example, people will understand "Data Science is super cool", but not "science super cool data is".  Likewise, the computer didn't understand print(55 + 5) print(300 - 8) print(21 + 67) because the syntax was wrong.
# MAGIC 
# MAGIC **Running into errors while programming is more common than you would think!** Forget those action scenes from blockbuster movies where hackers code faster than the speed of light! The real life of a programmer often times is about examining (calmly and with patience) why a certain error occurred.

# COMMAND ----------

# MAGIC %md
# MAGIC After learning about some syntax rules, let's dive into the next task.
# MAGIC 
# MAGIC ### Task 1.1.3:
# MAGIC 1. Try to run the instructions in the line below and see what the computer outputs as result. Remember that each command must be on a separate line.
# MAGIC 
# MAGIC - print(30 + 20 + 50)
# MAGIC - print(5)
# MAGIC - print(-2)

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC In the line above, three instructions were given to the computer:
# MAGIC 
# MAGIC - print(30 + 20 + 50)
# MAGIC - print(5)
# MAGIC - print(-2)
# MAGIC 
# MAGIC All these instructions are collectively known as **code**. Each line of command is known as a **line of code**. When we program or when we write code, we instruct or program the computer to do some tasks. Therefore, we can also call the code we write a computer program, or just simply a program.
# MAGIC 
# MAGIC The program we wrote in *Task 2.1* had three lines of code, but a program can be as small as one line.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.1.3.2:
# MAGIC 1. Try to use the `print()` command and write a program that consists of three lines of code:
# MAGIC - displays the result of 58 + 6
# MAGIC - displays the number 21
# MAGIC - displays the number -21
# MAGIC 2. Hit the Run button after you have finished and compare your output with our task requirement.

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Code Comments
# MAGIC Before we get down to real programming, we want to introduce you the # symbol, also known as the **comment symbol**. 
# MAGIC Any code or characters that follows the # symbol is called a code comment. Programmers use this symbol to add information or comments about our code.

# COMMAND ----------

print(3 + 1)
print(10 - 9) #This is the line that outputs 1

# COMMAND ----------

# MAGIC %md
# MAGIC We can also use code comments to add a general description at the beginning of our program.
# MAGIC Now let's have some practice with the ``#`` symbol.
# MAGIC Please type the following code in the editor below:
# MAGIC - #print(5 + 21) <br>
# MAGIC - #print(5) <br>
# MAGIC - #print(-5) <br>

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC Uncomment these three lines of code by removing the ``#`` symbol and then click the Run button.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Arithmetical Operations
# MAGIC In the previous section, we have performed additions and subtractions. To perform multiplication, we can use the * symbol. To multiply 5 * 2 we can type:

# COMMAND ----------

5 * 2 

# COMMAND ----------

# MAGIC %md
# MAGIC To perform division, we can use the / symbol. To divide 10 / 5:

# COMMAND ----------

10 / 5

# COMMAND ----------

# MAGIC %md
# MAGIC To raise a number to a power, we can use **. For example, to raise 2 to the power of 3:

# COMMAND ----------

3**3 

# COMMAND ----------

# MAGIC %md
# MAGIC The arithmetical operations in Python follow the usual order of operations from mathematics. Just like from mathematics, calculations in parentheses are calculated first, then follows by exponentiation, division, multiplication, and at last, addition and subtraction.
# MAGIC 
# MAGIC ### Task 1.1.5: 
# MAGIC 1. Perform the following calculations in Python.
# MAGIC - 5 * 30
# MAGIC - 56 / 2
# MAGIC - 5^2

# COMMAND ----------

# Start your code below:


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC You have just completed your first training notebook.
