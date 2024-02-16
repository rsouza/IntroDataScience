# Databricks notebook source
# MAGIC %md
# MAGIC ## Object Oriented Programming (OOP)
# MAGIC
# MAGIC There are only two options on how I think you will perceive this lecture. If you have some IT background and already some experience with OOP, this notebook will be a piece of cake. If these notebooks are your first programming experience, you may feel a bit confused at the beginning. Don't worry, once you get a grip of it, the path of OOP in Python will be smooth! The truth is, we have all been there when we first met OOP. 
# MAGIC
# MAGIC ## What is OOP?
# MAGIC Imagine that you have a store with clothes. You are an IT person which is in charge of software that is supporting this store. The software holds information on all the products that are being offered - prices, sizes, counts, colours, anything. As we are talking about clothes - new pieces of clothing are coming every month. During the first months, you are describing with data structures manually every piece of clothing - what its parameters are, what functions are available in the online shop with regards to this item.
# MAGIC
# MAGIC After a few months, you get bored by how repetitive your work is! You realize that many items belong into a certain **group**. For example, many items can be simply regarded as *shirts*. Now, each shirt is going to have some **characterists** and certain **actions** can be done with it. For example, it has collar, it covers upper part of body. We are able to filter by the general size, size at waist and also length within our online shop. Each particular shirt is an **object**. The *shirt group* is a  **class** into which all particular objects (shirts) belong. This class is going to have some **methods** which relate to the characteristics of the shirts and what can be done with those. Each particular shirt which is in the store will now **inherit** some properties of the class, which will simplify the job for us.
# MAGIC
# MAGIC The creation of classes, with respective methods and inheritance for objects will save us a lot of time! **Whenever a new shirt arrives, we just attribute it into shirts class and so it will instantly inherit all the methods.**
# MAGIC
# MAGIC ***

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Classes and Objects
# MAGIC
# MAGIC To begin with this notebook, let's start by looking at the following example:

# COMMAND ----------

l = [1, 2, 3]
s = "string"
d = {"a": 1, "b": 2}

# COMMAND ----------

# MAGIC %md
# MAGIC The code above have variables that consists of various classes. If we use the `type()` function, it will tell us everything:

# COMMAND ----------

print(type(l))
print(type(s))
print(type(d))

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that when we used the `type()` function, the values labeled "class" is returned. We can deduct that "type" and "class" are used interchangeably. You see that we've been using classes for some time already:
# MAGIC
# MAGIC - Python lists are objects of the <b>list</b> class.
# MAGIC - Python strings are objects of the <b>str </b>class.
# MAGIC - Python dictionaries are objects of the <b>dict</b> class.
# MAGIC
# MAGIC In this chapter, we will go on a journey of creating a class of our own. We're going to create a simple class called `NewList` and recreate some of the basic functionality of the Python list class.
# MAGIC
# MAGIC Before we get started, let's look at the relationship between objects and classes.
# MAGIC
# MAGIC - An <b> object</b> is an entity that stores data.
# MAGIC - An object's <b>class</b> defines specific properties objects of that class will have.
# MAGIC
# MAGIC > A class is a template for objects. A class defines object properties including a valid range of values, and a default value. A class also describes object behavior. 
# MAGIC
# MAGIC > An object is a member or an "instance" of a class. 
# MAGIC An object has a state in which all of its properties have values that you either explicitly define or that are defined by default settings.
# MAGIC
# MAGIC You can read more about classes and objects here: https://www.javatpoint.com/python-oops-concepts.
# MAGIC For people who previously have little programming experience, we highly recommend you to watch this short video explaining [what is object oriented programming](https://www.youtube.com/watch?v=xoL6WvCARJY).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Defining a Class
# MAGIC
# MAGIC Now the question arises of how we can define a class in Python. It turns out that creating a class is very similar to how we define a function.
# MAGIC
# MAGIC Note that we are using the ``pass`` keyword. The `pass` keyword is used as a placeholder for future code in code blocks where code is required syntactically like a class, function, method, for loop or if statement. If we were to not use the ``pass`` keyword an error would be thrown.

# COMMAND ----------

#This is a function:
def my_function():
    # the details of the
    # function go here
    pass
    
#This is a class:
class MyClass():
    # the details of the
    # class go here
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC Similar to a function, parentheses and a colon are used after the class name ``():`` when defining a class. Just like a function, the body of our class is indented like a function's body is.
# MAGIC
# MAGIC The rules for naming classes are the same as they are for naming functions and variables.
# MAGIC There is a **general rule of thumb in naming functions and classes**: 
# MAGIC - when naming for variables and and **functions**, all lowercase letters are used with underscores between: `like_this` 
# MAGIC - And when naming **classes**, there are no underscores are used between words, and the first letter of each word is capitalized: `LikeThis` 
# MAGIC
# MAGIC Following is an example of a definition of a class named ``MyClass``:
# MAGIC ````python
# MAGIC class MyClass():
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.3.2.1:
# MAGIC Define a new class named `NewList()`. Remember to use the `pass` keyword in the body of our class to avoid a SyntaxError.

# COMMAND ----------

#Start your code below:

class NewList():
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC In OOP (Object-oriented programming), we use <b> instances </b> to describe each different object. Let's look at an example:

# COMMAND ----------

#These objects are two instances of the Python str class.
string_1 = "The first string"

string_2 = "The second string"

#While each is unique - they contain unique values - they are the same type of object.

# COMMAND ----------

# MAGIC %md
# MAGIC Once we have defined our class, we can create an object of that class, which is known as **instantiation**. If you create an object of a particular class, the technical phrase for what you did is to "instantiate an object of that class." Let's learn how to instantiate an instance of our new class:
# MAGIC
# MAGIC ````python
# MAGIC my_class_instance = MyClass()
# MAGIC ````
# MAGIC
# MAGIC This single line performed two thigns:
# MAGIC - Instantiation of an object of the class `MyClass`.
# MAGIC - Assignment of this instance to the variable named `my_class_instance`.
# MAGIC
# MAGIC To illustrate this more clearly, let's look at an example using Python's built-in integer class. In the previous mission, we used the syntax `int()` to convert numeric values stored as strings to integers. Let's look at a simple example of this in code and break down the syntax into parts, which we'll read right-to-left:
# MAGIC
# MAGIC ````python
# MAGIC my_int = int("5")
# MAGIC ````

# COMMAND ----------

# MAGIC %md
# MAGIC To break this down:
# MAGIC - `int("5")` <<< Instantiate an object of the class int
# MAGIC - `my_int` <<< Assign the object to a variable with the name `my_int`
# MAGIC >The syntax to the right of the assignment operator ``=`` **instantiates** the object, and the assignment operator and variable name create the variable. 
# MAGIC
# MAGIC This helps us understand some of the subtle differences between an object and a variable.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.3.2.2:
# MAGIC 1. Define a new class called `NewList`:
# MAGIC     - Use `NewList()` when defining the class.
# MAGIC     - Use the pass keyword so our empty class does not raise a `SyntaxError`.
# MAGIC 2. Create an instance of the `NewList` class. Assign it to the variable name `newlist_1`.
# MAGIC 3. Print the type of the `newlist_1` variable.

# COMMAND ----------

# Start your code below:

class NewList():
    pass

newlist_1 = NewList()
print(type(newlist_1))

# COMMAND ----------

# MAGIC %md
# MAGIC Lovely! We have just created and instantiated our fist class! However, our class is lacking some of behaviours, it doesn't do anything yet. 
# MAGIC That means we need to define some **methods** which allow objects to perform actions.
# MAGIC
# MAGIC Let's think of methods like special functions that belong to a particular class. This is why we call the replace method `str.replace()` — because the method belongs to the str class.
# MAGIC
# MAGIC While a function can be used with any object, did you know that each class has its own set of methods? Let's look at an example using some Python built in classes:

# COMMAND ----------

my_string = "hello world"   # an object of the str class
my_list = [2, 4, 8]   # an object of the list class

# COMMAND ----------

# MAGIC %md
# MAGIC All list objects have the `list.append()` method. Let's try this:

# COMMAND ----------

my_list.append(4)
print(my_list)

# COMMAND ----------

# MAGIC %md
# MAGIC Also, we have learned the `str.replace()` method in our previous chapter. This method belongs to the string class.

# COMMAND ----------

my_string = my_string.replace("h","H")
print(my_string)

# COMMAND ----------

# MAGIC %md
# MAGIC > **The interchanging of one method from one class to another class is forbidden in Python:**

# COMMAND ----------

my_string.append("!") # can't use a method from one class with the other class

# COMMAND ----------

# MAGIC %md
# MAGIC How can we **create a method**? It is almost identical to how we create a function with one exception: the method is indented within our class definition. See example below:

# COMMAND ----------

class MyClass():
    def greet():
        return "hello"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.3.2.3:
# MAGIC
# MAGIC 1. Define a new class called `NewList()`.
# MAGIC     - Use `NewList()` when defining the class, so we can perform answer checking on your class.
# MAGIC 2. Inside the class, define a method called `first_method()`.
# MAGIC 3. Inside the method, return the string "This is my first method".
# MAGIC 4. Create an instance of the `NewList` class. Assign it to the variable name `newlist`.

# COMMAND ----------

# Start your code below:

class NewList():
    def first_method():
        return "This is my first method"

newlist = NewList()
print(type(newlist))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Understading 'self'
# MAGIC On the previous paragraphs, we defined a class with a simple method, then created an instance of that class:

# COMMAND ----------

class NewList():
    def first_method():
        print("hello")

instance = NewList()

#Let's look at what happens when we call (run) that method:
instance.first_method()

# COMMAND ----------

# MAGIC %md
# MAGIC This error is a bit confusing. It says that one argument was given to `first_method()`, but when we called the method we didn't provide any arguments. It seems like there is a "phantom" argument being inserted somewhere.
# MAGIC
# MAGIC When we call the `first_method()` method belonging to the instance object, Python interprets that syntax and adds in an argument representing the instance we're calling the method on.
# MAGIC
# MAGIC We can verify that this is the case by checking it with Python's built-in str type. We'll use `str.title()` to convert a string to title case.

# COMMAND ----------

# create a str object
s = "MY STRING"

# call `str.title() directly
# instead of `s.title()`
result = str.title(s)
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's study the following class carefully:

# COMMAND ----------

class MyClass():
    def print_self(self):
        print(self)

mc = MyClass()

#Next, let's print the mc object so we can understand 
#what the object itself looks like when its printed:

print(mc)

# COMMAND ----------

#Lastly, let's call our print_self() method to see
#whether the output is the same as when we printed the object itself:

mc.print_self()

# COMMAND ----------

# MAGIC %md
# MAGIC The same output was displayed both when we printed the object using the syntax `print(mc)` and when we printed the object inside the method using `print_self()` — **which proves that this "phantom" argument is the object itself**!
# MAGIC
# MAGIC Technically, we can give this first argument — which is passed to every method — any parameter name we like. However, the **convention is to call the parameter `self`**. This is an important convention, as without it class definitions can get confusing.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.3.3:
# MAGIC
# MAGIC In the editor below:
# MAGIC 1. Modify the `first_method()` method by changing the argument to `self`.
# MAGIC 2. Create an instance of the NewList class. Assign it to the variable name ``newlist``.
# MAGIC 3. Call `newlist.first_method()`. Assign the result to the variable ``result``.

# COMMAND ----------

 class NewList():
    def first_method(self):
        return "This is my first method."
newlist = NewList()
result = newlist.first_method()
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Creating a Method That Accepts an Argument
# MAGIC
# MAGIC The method we worked with on the previous two screens didn't accept any arguments except the self argument. Like with functions, methods are often called with one or more arguments so that the method can use or modify that argument.
# MAGIC
# MAGIC Let's create a method that accepts a string argument and then returns that string. The first argument will always be the object itself, so we'll specify self as the first argument, and the string as our second argument:

# COMMAND ----------

class MyClass():
    def return_string(self, string):
        return string

# COMMAND ----------

# MAGIC %md
# MAGIC Let's instantiate an object and call our method. Notice how when we call it, we leave out the self argument, just like we did on the previous screen:

# COMMAND ----------

mc = MyClass()
result = mc.return_string("Hey there!")
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC Now it's time to pratice creating methods for our classes.
# MAGIC
# MAGIC ### Task 2.3.4:
# MAGIC 1. Define a new class called `NewList()`.
# MAGIC 2. Inside the class, define a method called `return_list()`.
# MAGIC     - The method should accept a single argument `input_list` when called.
# MAGIC     - Inside the method, return `input_list`.
# MAGIC 3. Create an instance of the NewList class, and assign it to the variable name `newlist`.
# MAGIC 4. Call the `newlist.return_list()` method with the argument [1, 2, 3]. Assign the result to the variable `result`.

# COMMAND ----------

# Start your code below:

class NewList():
    def return_list(self,input_list):
        return input_list
newlist = NewList()
result = newlist.return_list([1,2,3])
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Atrributes and the Init Method (IMPORTANT)
# MAGIC Let's recap what we've already learned since the beginning of this lecture.
# MAGIC
# MAGIC - We can define a ``class``.
# MAGIC - We know that a class can have ``methods``. These are something like a functions or commands which can be performed with the objects belonging to that class.
# MAGIC
# MAGIC We now need to learn about **2 new things** at the same time, as they are closely related:
# MAGIC
# MAGIC - init method
# MAGIC - attributes
# MAGIC
# MAGIC The power of objects is in their ability to store data, and data is stored inside objects using **attributes**. You can think of attributes like **special variables that belong to a particular class**. Attributes let us store specific values about each instance of our class. When we instantiate an object, most of the time we specify the data that we want to store inside that object. Let's look at an example of instantiating an int object:

# COMMAND ----------

my_int = int("3")

# COMMAND ----------

# MAGIC %md
# MAGIC When `int()` was used, the argument "3" was provided, which was converted and stored inside the object. **We define what is done with an arguments provided at instantiation using the init method.**
# MAGIC
# MAGIC > The init method — also called a ``constructor`` — is a special method that runs when an instance is created so we can perform any tasks to set up the instance.
# MAGIC
# MAGIC The init method has a **special name that starts and ends with two underscores: `__init__()`**. Let's look at an example:

# COMMAND ----------

class MyClass():
    def __init__(self, string):
        print(string)

mc = MyClass("Hallo!")

# COMMAND ----------

# MAGIC %md
# MAGIC Let me give you a step by step guide below:
# MAGIC    - defined the `__init__()` method inside our class as accepting two arguments: `self` and `string`.
# MAGIC    - Inside the `__init__()` method, the `print()` function on the `string` argument was called.
# MAGIC    - mc (our MyClass object) was instantiated, "Hallo!" was passed as an argument. The init function ran immediately, displaying the text "Hallo!"
# MAGIC     
# MAGIC The init method's most common usage is to store data as an attribute:

# COMMAND ----------

class MyClass():
    def __init__(self, string):
        self.my_attribute = string

mc = MyClass("Hallo!") # When we instantiate our new object, 
# Python calls the init method, passing in the object

# COMMAND ----------

# MAGIC %md
# MAGIC Our code didn't print any output, but "Hallo" was stored in the attribute `my_attribute` inside our object. Like methods, attributes are accessed using dot notation, but attributes don't have parentheses like methods do. Let's use dot notation to access the attribute:

# COMMAND ----------

print(mc.my_attribute)

# COMMAND ----------

# MAGIC %md
# MAGIC To summarize some of the differences between attributes and methods:
# MAGIC - An attribute is used to store data, very similar to an variable.
# MAGIC - A method is used to perform actions, very similar to a function.

# COMMAND ----------

# MAGIC %md
# MAGIC To summarize what we've learned so far:
# MAGIC
# MAGIC - The power of objects is in their ability to store data.
# MAGIC - Data is stored as attributes inside objects.
# MAGIC - We access attributes using dot notation.
# MAGIC - To give attributes values when we instantiate objects, we pass them as arguments to a special method called `__init__()`, which runs when we instantiate an object.
# MAGIC
# MAGIC We now have what we need to create a working version of our NewList class! This first version will:
# MAGIC
# MAGIC - Accept an argument when you instantiate a NewList object.
# MAGIC - Use the init method to store that argument in an attribute: `NewList.data`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.3.5:
# MAGIC
# MAGIC 1. Define a new class called `NewList()`.
# MAGIC 2. Create an init method which accepts a single argument, `initial_state`.
# MAGIC 3. Inside the init method, assign `initial_state` to an attribute called `data`.
# MAGIC 4. Instantiate an object of your NewList class, providing the list [1, 2, 3, 4, 5] as an argument. Assign the object to the variable name `my_list`.
# MAGIC 5. Use the `print()` function to display the data attribute of `my_list`.

# COMMAND ----------

# Start your code below:

class NewList():
    def __init__ (self,initial_state):
        self.data = initial_state

my_list = NewList([1,2,3,4,5])
print(my_list.data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Creating and Updating an Attribute (OPTIONAL)
# MAGIC
# MAGIC To summarize the work we've done so far:
# MAGIC - We've created a <b>NewList class</b> which stores a list at the point of instantiation using the init constructor.
# MAGIC - We stored that list inside an attribute `NewList.data`.
# MAGIC     
# MAGIC Now we want to add some new functionality: a new attribute. 

# COMMAND ----------

# MAGIC %md
# MAGIC When we want to find the length of a list, we use the `len()` function.
# MAGIC What if we created a new attribute, `NewList.length`, which stores the length of our list at all times? We can achieve this by adding some to the init method:

# COMMAND ----------

class NewList():
    """
    A Python list with some extras!
    """
    def __init__(self, initial_state):
        self.data = initial_state

        # we added code below this comment
        length = 0
        for item in self.data:
            length += 1
        self.length = length
        # we added code above this comment

    def append(self, new_item):
        """
        Append `new_item` to the NewList
        """
        self.data = self.data + [new_item]

# COMMAND ----------

# MAGIC %md
# MAGIC Let's have a closer look at what happens when we use the `NewList.length` attribute as defined above:

# COMMAND ----------

my_list = NewList([1, 2, 3])
print(my_list.length)

my_list.append(4)
print(my_list.length)

# COMMAND ----------

# MAGIC %md
# MAGIC Because the code we added that defined `NewList.length` was added **only in the init method, if the list is made longer using the `append()` method, our `NewList.length` attribute is no longer accurate.**
# MAGIC
# MAGIC To address this, we need to run the code that calculates the length after any operation which modifies the data, which, in our case, is just the `append()` method.
# MAGIC
# MAGIC Rather than writing the code out twice, we can add a helper method, which calculates the length, and just call that method in the appropriate places.
# MAGIC
# MAGIC Here's a quick example of a helper method in action:

# COMMAND ----------

class MyBankBalance():
    """
    An object that tracks a bank
    account balance
    """

    def __init__(self, initial_balance):
        self.balance = initial_balance
        self.calc_string()

    def calc_string(self):
        """
        A helper method to update self.string
        """
        string_balance = "${:,.2f}".format(self.balance)
        self.string = string_balance

    def add_value(self, value):
        """
        Add value to the bank balance
        """
        self.balance += value
        self.calc_string()

mbb = MyBankBalance(3.50)
print(mbb.string)

# COMMAND ----------

# MAGIC %md
# MAGIC In the code above, a helper method `MyBankBalance.calc_string()` was created. This method calculate a string representation of object's bank balance stored in the attribute `MyBankBalance.string`. We called that helper method from the init method so it updates based on the initial value.

# COMMAND ----------

# MAGIC %md
# MAGIC Another helper method from the `MyBankBalance.add_value()` method was called, so the value updates whenever the balance is increased: 

# COMMAND ----------

mbb.add_value(17.01)
print(mbb.string)

# COMMAND ----------

mbb.add_value(5000)
print(mbb.string)

# COMMAND ----------

# MAGIC %md
# MAGIC We see that our helper methods are defined after our init method. We mentioned earlier that the order in which you define methods within a class doesn't matter, but there is a convention to order methods as follows:
# MAGIC
# MAGIC 1. Init method
# MAGIC 2. Other methods
