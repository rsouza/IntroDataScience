# Databricks notebook source
# MAGIC %md
# MAGIC # Visualizations with Pandas

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Introduction
# MAGIC 
# MAGIC Pandas library will probably be your first option to create visual insights to understand the data more.
# MAGIC The main advantage is that you can visualize the data using simple and straightforward methods. Behind the scenes of plotting with Pandas is another library - Matplotlib. But don't worry now about Matplotlib as we will cover it in the later lesson. So when we call some Pandas' plotting function, Matplotlib acts as an engine. Therefore we can use a higher level of code to gain similar good-looking plots as in Matplotlib.

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Dataset
# MAGIC 
# MAGIC We will be working with an occupancy detection dataset, that can be found [here](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+#). 
# MAGIC 
# MAGIC This dataset is intended to solve binary classification problems according to attributes which could indicate whether a person is present in the room or not.
# MAGIC 
# MAGIC Let´s import Pandas library and load the data.

# COMMAND ----------

# Import Pandas and Numpy libraries
import pandas as pd
import numpy as np
# For now, we use Matplotlib only to set the size of plots
import matplotlib.pyplot as plt
# rcParams allow us to set the size global to the whole notebook 
plt.rcParams['figure.figsize'] = [10,6]
# In some plots, we define the size within the plotting function

# COMMAND ----------

# Load the data
data = pd.read_csv('../Data/room_occupancy.txt')

# COMMAND ----------

# Take a look at the first 10 rows
data.head(10)

# COMMAND ----------

# Print number of rows and columns
data.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Attributes explanation
# MAGIC 
# MAGIC Firstly, let's examine what variables we are dealing with.
# MAGIC 
# MAGIC - date: the specific day and time when the values were recorded 
# MAGIC - Temperature: measured in Celsius
# MAGIC - Humidity: relative humidity - a present state of absolute humidity relative to a
# MAGIC     maximum humidity given the same temperature expressed as a          percentage
# MAGIC - Light: in Lux unit
# MAGIC - CO2: in ppm unit (parts per million)
# MAGIC - HumidityRatio: derived quantity from temperature and relative humidity, expressed in kilograms of water vapor per kilogram of dry air  
# MAGIC - Occupancy: 
# MAGIC the presence of a person in the room: the occupancy of the room was obtained from pictures that were taken every minute for a period of 8 days (1 if a person is present, 0 otherwise)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1.2 Exploring the data
# MAGIC Let´s look at the variables data type using `.dtypes` attribute.

# COMMAND ----------

# Check variables data type
data.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC There are 6 numerical variables in the dataset. Almost all of the variables have been correctly interpreted according to theiy datatypes, except for 'date' variable, which Python recognized as an object. Convert this object datatype to datetime datatype using `Pandas`. 

# COMMAND ----------

# Convert date variable using Pandas to_datetime method
data['date'] = pd.to_datetime(data['date'])

# COMMAND ----------

# MAGIC %md
# MAGIC Let´s check whether there are some missing values that we need to be aware of. 

# COMMAND ----------

# Explore descriptive statistics
data.describe()

# COMMAND ----------

# Check missing values
data.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Plot method
# MAGIC 
# MAGIC `.plot()` function
# MAGIC 
# MAGIC This plotting function is simply a wrapper around `Matplotlibs` plot function that create **a lineplot** by default. A lineplot plots each data point of DataFrame and then draw a straight continuous line connecting these values.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC If you call `.plot()` right onto the DataFrame, all of the variables will be plotted. This approach is sometimes not quite good as you can see below. There are several variables with similar low values, thus they are overlapping. 

# COMMAND ----------

# Create a lineplot of DataFrame
data.plot()

# COMMAND ----------

# MAGIC %md
# MAGIC Let´s create a violet line plot of variable 'Temperature'. You can select desired variable by indexing the DataFrame with square brackets. 

# COMMAND ----------

# Create a line plot of Temperature variable 
data['Temperature'].plot(color = 'violet')

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can see plotted only Temperature variable with respective values.  
# MAGIC 
# MAGIC You can see that `plot()` function returns ```<AxesSubplot:>``` object. But what does it mean? 
# MAGIC 
# MAGIC For now, remember that each graph is actually represented as a Figure object that serves as the base. Onto this bse, there is Axes object in which are x-axis and y-axis created. In the latter section, we´ll discuss more details about Figures and Axes. 
# MAGIC To avoid displaying this label, you can put a semicolon at the end of the line.
# MAGIC 
# MAGIC ---
# MAGIC It is also posible to plot multiple columns by passing a **list** of respective variables, separated by a comma within square brackets and then call `.plot()`. `Pandas` set the colors of lines automatically and so you will be able to distinguish the variables. You can specify colors of lines to `color` parameter if you want to. Chosen colors need to passed in the dictionary.  See the following example:
# MAGIC 
# MAGIC ``` data[['variable_1', 'variable_2']].plot(color = {'variable_1':'yellow', 'variable_2': 'black'})```
# MAGIC 
# MAGIC Also the legend is placed by default. 
# MAGIC 
# MAGIC **Try it yourself in the following task.**

# COMMAND ----------

# TASK 1 >>> Create a lineplot of variables 'Temperature' and 'Humidity'
#        >>> Set the color of Temperature to green and Humidity to be blue
#        >>> In the created plot, observe how humidity and temperature have been decreasing and increasing

data[['Temperature','Humidity']].plot(color = {'Temperature':'green', 'Humidity':'blue'});

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 Customizing parameters of a line plot

# COMMAND ----------

# MAGIC %md
# MAGIC It is possible to plot one column against another one, as we see in the example below. We specify date variable on the x-axis and CO2 variable on the y-axis.
# MAGIC Within `.plot()` method, you can set several parameters such as title, axis labels, size of plot, etc. For more information about parameter settings, see [the documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html)

# COMMAND ----------

# A lineplot of amount of CO2 per date

data.plot(x = 'date', y = 'CO2',             # Specifying variabes on the axis
          figsize = (15,8),                  # Setting a Figure object size as a tuple
          fontsize = 9,                      # Setting ticks font size 
          color = 'skyblue',                 # Setting a color
          title = 'Amount of CO2 over time', # Setting a title of a plot 
          xlabel = 'Date',                   # Customizing x-axis label (variable name by default)
          ylabel = 'CO2 (in ppm)');          # Customizing y-axis label (no label by default)

# COMMAND ----------

# MAGIC %md
# MAGIC The line plot is showing a trend of CO2 amount over a period of time from 12.02.2015 (Thursday) till 18.02.2015 (Wednesday). The graph displays how the amount of CO2 has decreased during the weekend (14.02.2015 - 15.02.2015). 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Plotting approaches and plot kinds
# MAGIC 
# MAGIC Before we proceed to the other plot kinds, there is an important thing to mention. 
# MAGIC 
# MAGIC **There are different ways how you can plot your data**. To not to get confused later on, let´s clarify them a little bit. 
# MAGIC 
# MAGIC In the preceding examples, we use the default `plot()` function that creates **a line plot by default**. 
# MAGIC 
# MAGIC **First approach:**
# MAGIC 
# MAGIC - there is a possibility to select other plot style you wich to create using `kind` parameter. 
# MAGIC 
# MAGIC Overall, there are 10 plot styles you can specify as an argument provided as a string:
# MAGIC 
# MAGIC - `hist`   - histogram  
# MAGIC - `box`    - boxplot  
# MAGIC - `bar`    - vertical barplot  
# MAGIC - `barh`   - horizontal barplot  
# MAGIC - `scatter`- scatterplot  
# MAGIC - `pie`    - pie plot  
# MAGIC - `kde`    - density plot  
# MAGIC - `density`- density plot  
# MAGIC - `area`   - area plot  
# MAGIC - `hexbin` - hexagonal bin plot  
# MAGIC 
# MAGIC **Second approach:**
# MAGIC 
# MAGIC - all of these plots can be created using corresponding plotting functions:
# MAGIC 
# MAGIC - `DataFrame.plot.line`
# MAGIC - `DataFrame.plot.hist`
# MAGIC - `DataFrame.plot.box`
# MAGIC - `DataFrame.plot.bar`
# MAGIC - `DataFrame.plot.barh`
# MAGIC - `DataFrame.plot.scatter`
# MAGIC - `DataFrame.plot.pie`
# MAGIC - `DataFrame.plot.kde`
# MAGIC - `DataFrame.plot.density`
# MAGIC - `DataFrame.plot.area`
# MAGIC - `DataFrame.plot.hexbin`
# MAGIC 
# MAGIC For a histogram and a boxplot there are aditionally two plotting functions: `DataFrame.hist()` and `DataFrame.boxplot()`.
# MAGIC 
# MAGIC It is up to you which of the two approaches you will decide to stick with.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2.1 Histogram

# COMMAND ----------

# MAGIC %md
# MAGIC Histogram is a handy plot to get a picture of the spread of data points. It creates so-called bins that you can think of as particular ranges of values. Each data point falls into the respective bin according to its value and the number of data points are counted. 
# MAGIC 
# MAGIC Look at some statistics computed on variable 'HumidityRatio. We know the lowest and the highest ratio, average value, and that 50% of values are under 30.0045, etc. But it´s hard to imagine how the distribution of 9752 observations looks like based these summary statistics.

# COMMAND ----------

# Take a look at the descriptive statistics using .describe()
data.HumidityRatio.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Let´s create a histogram of variable 'HumidityRatio'. You can use `.plot()` method and specify `kind = "hist"` within this method. Again, we select desired column by indexing the DataFrame. In this case, `Pandas` do not create x-axis label. Since `plot` function returns `matplotlib.axes.AxesSubplot`object, we have access to Matplotlib capabilities and can specify labels.
# MAGIC 
# MAGIC `plt.xlabel` for x-axis  
# MAGIC `plt.ylabel` for y-axis

# COMMAND ----------

# The Humidity ratio
data['HumidityRatio'].plot(kind ='hist',
                      figsize = (11,8),
                      color = '#ff7f0e',
                      alpha = .5,                            # Setting the transparency of a color                
                      title = 'Humidity ratio distribution')

plt.xlabel('Humidity Ratio');                                # Creating x-axis label

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can see all of the 9752 data points plotted in one graph. In the resulting graph, we can see how the values are spread across a range from about 0.003 to 0.005. Most data points lie in the range of values 0.0043 to 0.0047.

# COMMAND ----------

# MAGIC %md
# MAGIC Be default, this method separate all of the records into 10 bins. To verify whether our resulting graph accurately reflects the underlying data we should have some fun a and try to use different number of bins. Using different number of bins, the visual appearance of the histogram will be changing.

# COMMAND ----------

# TASK 2 >>> Create a histogram of 'Humidity' in the same way as above
#        >>> Try yourself to change the number of bins and observe the output 
#        >>> Play around and change the size of plot and the transparency to see the differences
#        >>> Set x-axis label to 'Relative Humidity (%)'
#        >>> Change default y-axis label (Frequency) to 'Number of occurences'

data['Humidity'].plot(kind ='hist',
                      figsize = (11,8),
                      bins = 15,
                      color = '#ff7f0e',
                      alpha = .5,
                      title = 'Humidity distribution')
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Number of occurences');

# COMMAND ----------

# MAGIC %md
# MAGIC By applying `.hist()` onto the DataFrame, we are able to create a histogram of selected column or columns. It is also posibble to visualize a distribution of variables of the whole dataset.

# COMMAND ----------

# The data distribution of the whole DataFrame

data.hist(layout=(2,4), grid = False)                    # Setting layout of 2 rows and 4 column and Disabling the grid
plt.suptitle('The distribution of variables');           # Adding a suptitle using Matplotlib

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2.2 Boxplot (box-and-whisker plot)
# MAGIC 
# MAGIC Another approach to visualize the distribution of the data is using boxplots. Moreover, the resulting plot will include [five-number summary](https://en.wikipedia.org/wiki/Five-number_summary): the smallest data point value, the first quartile (1Q), the median (Q2), the third quartile (Q3) and the largest data point value. Boxplots can reveal to us whether there are some possible outliers in the DataFrame and whether the data distribution is skewed or rather symmetrical.
# MAGIC 
# MAGIC In order to draw a boxplot you can use `.plot()` method with specifying the `kind` keyword argument as 'box'. If we want to customize things such as color, width or style of the line, we can use `boxprops`,`whiskerprops`,`medianprops` and `capprops` parameter.

# COMMAND ----------

# A boxplot of relative humidity

data['HumidityRatio'].plot(kind = 'box', 
                           figsize = (9,5),
                           boxprops = dict(linewidth = 1.5, color = 'green',linestyle = '-.'),# Customizing the box
                           whiskerprops = dict(linewidth = 1.5, color = 'pink'),              # Customizing the whiskers
                           medianprops = dict(linewidth = 1.5, color = 'red'),                # Customizing median line
                           capprops = dict(linewidth = 1.5, color = 'darkblue'),              # Customizing caps on the whiskers
                           title = 'Boxplot of Humidity ratio');

# COMMAND ----------

# MAGIC %md
# MAGIC This boxplot illustrates how the values of humidity ratio are spread out. Based on the shape it seems that distribution is rather symetrical and also there are no extreme values, e.g. outliers. The actual box represent 50% of records along with the median value that is displayed as a red line. You can return actual values (of quartiles/percentiles) using `quantile` function.

# COMMAND ----------

# Compute 25th percentile, median value and 75th percentile of HumidityRatio variable
perc_25, median, perc_75 = data.HumidityRatio.quantile([.25,.5,.75])

# COMMAND ----------

# Print the output
perc_25, median, perc_75

# COMMAND ----------

# MAGIC %md
# MAGIC The alternative is to use `pandas` built-in method `DataFrame.boxplot()`. Since the boxplots are really usefull when comparing two or more groups, we´ll look at the amount of carbon dioxide according to a person´s presence. Selecting groups you´d like to compare can be done using `by` parameter with the respective variable. We´ll adjust the Figure size and rename x-tick labels.
# MAGIC 
# MAGIC The axis grid lines are displayed by default. You can disable showing these lines by setting parameter `grid = False`.

# COMMAND ----------

# A boxplot of CO2 by occupancy of the room

data.boxplot(figsize = (10,8),
             column = 'CO2', 
             by = 'Occupancy')

# Setting x-tick labels using Matplotlib
plt.xticks([1,2], ['Not occupied room','Occupied room']);

# COMMAND ----------

# MAGIC %md
# MAGIC Comparing the distribution of two groups can helps us to better understand the data. From the boxplots you can imediately see the difference in amount of CO2. When the room is occupied, the amount of carbon dioxide is higher, while 50% of data points have values in the range of about 620 up to slightly above 1000 ppm. Also the median values are completely different. When the room is empty, the amount of carbon dioxide is substantially lower, although plot indicates a lot of outliers. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2.3 Bar plot
# MAGIC 
# MAGIC Bar plots are useful when we want to compare categorical data based on their values. Each of the category will be plotted on the x-axis and the height of bars will be presented corresponding values. 
# MAGIC 
# MAGIC Let´s create simple DataFrame of students and theirs scores from exam for illustration. 

# COMMAND ----------

# Example data

sample_data = pd.DataFrame({'Student': ['Thomas','Margaret','Lisa','John','Elis','Sally','Marc','Angela','Sebastian'],
                            'Score': [78,50,68,83,99,98,65,90,85],
                            'Class': ['A','B','B','B','A','A','A','B','A']})
sample_data

# COMMAND ----------

# A vertical barplot of students by their score

sample_data.sort_values(by = 'Score', ascending = False).plot(x = 'Student', 
                                                              y = 'Score',
                                                              kind = 'bar', 
                                                              rot = 45,
                                                              color = ['mediumseagreen','lightgreen','sandybrown',
                                                                      'lightcoral','wheat','lightsteelblue',
                                                                      'slategrey','teal','black'],
                                                              legend = False)
plt.ylabel('Count');

# COMMAND ----------

# MAGIC %md
# MAGIC For creating a bar plot, we sort values by score in descending fashion to display student´s score. If your categories have longer labels, it´s appropriate to set rotation in order to avoid overlapping. You can explicitly set color of each bar either through specifying color name or hexadecimal color codes to `color` parameter.
# MAGIC 
# MAGIC See this [link](https://seaborn.pydata.org/tutorial/color_palettes.html) from Seaborn where the general principles of using color in plots are described. 
# MAGIC ___
# MAGIC 
# MAGIC You can also choose one of the built-in colormaps provided by Matplotlib. Colormaps can be accessed through `plt.cm` (`cm` stands for colormap). After that, specify a chosen colormap by its name. A reversed version of each available colormap can be done by appending `_r`to colormap´s name. Using Numpy´s `arange()` function we specify an interval of colors we want to select. 

# COMMAND ----------

# TASK 3 >>> Reuse the code above and create a horizontal barplot (kind = 'barh') of students score
#        >>> Set the colormaps with code: plt.cm.Set3_r(np.arange(len(sample_data)))
#        >>> Disable a legend
#        >>> Set x-axis label to 'Count'

sample_data.sort_values(by = 'Score').plot(x = 'Student', 
                                           y = 'Score',
                                           kind = 'barh', 
                                           color = plt.cm.Set3_r(np.arange(len(sample_data))),
                                           legend = False)
plt.xlabel('Count');

# COMMAND ----------

# MAGIC %md
# MAGIC To visualize count of students based on in which class they are, we count the number of occurences and plot it.

# COMMAND ----------

sample_data.Class.value_counts().plot(kind = 'bar', rot = 0)
plt.xlabel('Class')
plt.ylabel('Count');

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Practice time
# MAGIC 
# MAGIC For the further visualizations we´ll use data related to habits of individuals in terms of eating habits, transportation and devices they use and attributes of physical condition. This dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+#).

# COMMAND ----------

# Read the data
data_ob = pd.read_csv('../Datasets/obesity_data.csv')

# COMMAND ----------

# Take a look at the data
data_ob

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Attributes explanation
# MAGIC 
# MAGIC The first 5 variables involves some basic information about individuals along with the information about presence of overweight in a family.
# MAGIC 
# MAGIC - Gender: the gender
# MAGIC - Age: the age
# MAGIC - Height: the height 
# MAGIC - Weight: the weight 
# MAGIC - family_history_with_overweight: family member who is/was overweight
# MAGIC 
# MAGIC Variables related to eating habits:
# MAGIC 
# MAGIC - FAVC: whether an individual consumes high caloric food frequently
# MAGIC - FCVC: how frequently is vegetables used in main meals
# MAGIC - NCP: the number of main meals per day
# MAGIC - CAEC: consuming any food between main meals
# MAGIC - SMOKE: smoking
# MAGIC - CH2O: consumption of water per day
# MAGIC 
# MAGIC Variables related to physical condition:
# MAGIC - SCC: calories intake tracking
# MAGIC - FAF: physical activity frequency
# MAGIC - TUE: usage of technological devices per day
# MAGIC - CALC: alcohol consumption
# MAGIC - MTRANS: type of transportation
# MAGIC 
# MAGIC The last variable 'NObeyesdad' was created using equation for BMI (Body Mass Index) for each individual. Resulting values were compared with the data provided by WHO (World Health Organization) and the Mexican Normativity. 
# MAGIC 
# MAGIC Resulting labels:
# MAGIC 
# MAGIC -Underweight (< 18.5)   
# MAGIC -Normal (18.5 - 24.9)   
# MAGIC -Overweight (25 - 29.9)  
# MAGIC -Obesity I (30 - 34.9)  
# MAGIC -Obesity II (35 - 39)  
# MAGIC -Obesity III (> 40)  

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1.1 Exploring the data

# COMMAND ----------

# Check variables data type
data_ob.dtypes

# COMMAND ----------

# Explore descriptive statistics
data_ob.describe()

# COMMAND ----------

# Check missing values
data_ob.isnull().sum()

# COMMAND ----------

# Duplicated rows
data_ob.duplicated().sum()

# COMMAND ----------

# Removing duplicated rows
data_ob.drop_duplicates(inplace = True)

# COMMAND ----------

# TASK 4 >>> Create a boxplot of 'Age' variable
#        >>> Set the size of a plot to (10,8)

data_ob.boxplot(column = 'Age',
                figsize = (10,8));

# COMMAND ----------

# MAGIC %md
# MAGIC > What can you say about the distribution of ages? How old is the majority of people ? Is the distribution symmetrical; or are there people who are older compared to the majority?

# COMMAND ----------

# TASK 5 >>> Visualize the distribution of 'Weight' variable using a histogram
#        >>> Disable a grid line
#        >>> Try to set different numbers of bins

data_ob.Weight.hist(bins = 15, 
                    grid = False);

# COMMAND ----------

# MAGIC %md
# MAGIC > Based on the histogram, what is the most common weight? Does this distribution look symmetrical, oir rather bimodal? Could a different number of bins reflect the data better?

# COMMAND ----------

# TASK 6 >>> Create a normalized barplot of 'Gender' - specify parameter: normalize = True, to get percentages instead of counts
#        >>> Assing different colors to males and females (male to 'salmon', female to 'skyblue')
#        >>> Add title: 'Proportion of gender'
#                x-axis label: 'Gender' 
#                y-axis label: 'The percentage'

data_ob.Gender.value_counts(normalize = True).plot(kind = 'bar', 
                                                   color = ['salmon','skyblue'], 
                                                   title = 'Proportion of gender')
plt.xlabel('Gender')
plt.ylabel('The percentage');

# COMMAND ----------

# TASK 7 >>> Create boxplots of 'Age' by gender

data_ob.boxplot(column = 'Age', 
                by = 'Gender');

# COMMAND ----------

# MAGIC %md
# MAGIC > Compare the ranges of values for male and female: Is the distribution similar or is there some difference? Do you see any outliers? 

# COMMAND ----------

# TASK 8 >>> Find out how many people eat frequently high caloric meals ('FAVC')
#        >>> Set rotation of x-ticks to 0 degrees

data_ob.FAVC.value_counts().plot(kind = 'bar', 
                                 rot = 0);

# COMMAND ----------

# MAGIC %md
# MAGIC The question we might ask regarding obesity levels could be: 'Which individuals are more likely to be obese based on their age?' or 'Are younger adults overweight?'.
# MAGIC 
# MAGIC Firstly, let´s look how many levels are there. We call `.value_counts()` function on 'Nobeyesdad' variable and sort categories based on the obesity level.

# COMMAND ----------

# Counting values of the obesity levels
data_ob.NObeyesdad.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC Firstly, we define order of obesity levels in list and assign it to variable obesity_levels.

# COMMAND ----------

# Ordered categories
obesity_levels = ['Insufficient_Weight','Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II',
                  'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']

# COMMAND ----------

# MAGIC %md
# MAGIC Then we create CategoricalDtype where we specify categories and orderness.

# COMMAND ----------

# Creating CategoricalDtype 
ob_level = pd.api.types.CategoricalDtype(ordered = True, categories = obesity_levels)

# COMMAND ----------

# MAGIC %md
# MAGIC The last step is converting original data type of '.NObeyesdad' variable to created CategoricalDtype.

# COMMAND ----------

# Converting 'NObeyesdad' to created categoricalDtype
data_ob.NObeyesdad = data_ob.NObeyesdad.astype(ob_level)

# COMMAND ----------

# MAGIC %md
# MAGIC We can plot obesity levels to see how these vary according to age of individuals. To avoid overlapping label names of these categories, you can set the rotation of labels to 45 degree with `rot` parameter. Also disabling grid lines can help visualization to be more comprendious.

# COMMAND ----------

# TASK 9 >>> Create a boxplot of column 'Age' by 'NObeyesdad'
#        >>> Set a Figure size: (10,8)
#        >>> Set labels on x-axis to 45 using rot parameter
#        >>> Disable grid line

data_ob.boxplot(figsize = (10,8),
                column = 'Age', 
                by = 'NObeyesdad',
                rot = 45, 
                grid = False);

# COMMAND ----------

# MAGIC %md
# MAGIC > Looking at the created plot, can you see some pattern? At what ages people tend to suffer from some kind of obesity level? 

# COMMAND ----------

# TASK 10 >>> Create a bar plot of 'MTRANS' variable to find out how many people use certain kind of transportation
#         >>> Set labels on x-axis to 45 using rot parameter
#         >>> Set color: 'skyblue'
#         >>> What is the most popular kind of transportation? 

data_ob.MTRANS.value_counts().plot(kind='bar', 
                                   rot = 45, 
                                   color = 'skyblue');

# COMMAND ----------

# MAGIC %md
# MAGIC Let´s filter only those individuals who use an automobile for transportation; or walk. Then we´ll look at their weight.

# COMMAND ----------

# Filter only those rows where transportation kind is 'Automobile' and 'Walking'
# DataFrame.query will be explained more in the latter notebook, so don´t worry about it now
subset_transport = data_ob.query('MTRANS in ["Automobile","Walking"]')

# COMMAND ----------

# TASK 11 >>> Create a boxplot of newly created subset_transport DataFrame
#         >>> Set a Figure size: (10,8)
#         >>> Set 'Weight' to column parameter and 'MTRANS' to by parameter
#         >>> Set labels on x-axis to 45 using rot parameter
#         >>> Disable a grid line

subset_transport.boxplot(figsize = (10,8),
                         column = 'Weight', 
                         by = 'MTRANS',
                         rot = 45, 
                         grid = False);

# COMMAND ----------

# MAGIC %md
# MAGIC > What can you say about the distribution of these two groups of people? Which group has lower weight overall? Looking at the boxplots, I think it´s kind of motivation for all of us to consider type of transportation next time we´ll need to go somewhere :) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Citation request:
# MAGIC Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models. Luis M. Candanedo, VÃ©ronique Feldheim. Energy and Buildings. Volume 112, 15 January 2016, Pages 28-39.
# MAGIC 
# MAGIC Palechor, F. M., & de la Hoz Manotas, A. (2019). Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico. Data in Brief, 104344.
# MAGIC 
# MAGIC Some material adapted for RBI internal purposes with full permissions from original authors. [Source](https://github.com/zatkopatrik/authentic-data-science) 
