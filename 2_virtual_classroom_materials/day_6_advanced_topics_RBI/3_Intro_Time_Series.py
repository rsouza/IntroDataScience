# Databricks notebook source
# MAGIC %md
# MAGIC ### Introduction to Time Series Analysis and Forecast
# MAGIC 
# MAGIC Inspired by [this](https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/), [this](https://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/) and [this](https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/) blog posts  

# COMMAND ----------

import os
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from IPython.display import display, Image

%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 8

# COMMAND ----------

# MAGIC %md
# MAGIC Time Series (referred as TS from now) is considered to be one of the less known skills in the analytics space. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Out journey would go through the following steps:  
# MAGIC 
# MAGIC + What makes Time Series Special?  
# MAGIC + Loading and Handling Time Series in Pandas  
# MAGIC + How to Check Stationarity of a Time Series?  
# MAGIC + How to make a Time Series Stationary?  
# MAGIC + Forecasting a Time Series  

# COMMAND ----------

# MAGIC %md
# MAGIC ### What makes Time Series Special?
# MAGIC 
# MAGIC As the name suggests, TS is a collection of data points collected at constant time intervals. These are analyzed to determine the long term trend so as to forecast the future or perform some other form of analysis. But what makes a TS different from say a regular regression problem? There are 2 things:
# MAGIC 
# MAGIC + It is time dependent. So the basic assumption of a linear regression model that the observations are independent doesn’t hold in this case.  
# MAGIC + Along with an increasing or decreasing trend, most TS have some form of seasonality trends, i.e. variations specific to a particular time frame. For example, if you see the sales of a woolen jacket over time, you will invariably find higher sales in winter seasons.  
# MAGIC 
# MAGIC Because of the inherent properties of a TS, there are various steps involved in analyzing it. These are discussed in detail below. Lets start by loading a TS object in Python. We’ll be using the popular AirPassengers data set which can be downloaded [here](https://www.analyticsvidhya.com/wp-content/uploads/2016/02/AirPassengers.csv).  
# MAGIC 
# MAGIC Please note that the aim of this notebook is to familiarize you with the various techniques used for TS in general. The example considered here is just for illustration and I will focus on coverage a breadth of topics and not making a very accurate forecast.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading and Handling Time Series in Pandas  
# MAGIC 
# MAGIC Pandas has dedicated libraries for handling TS objects, particularly the datatime64[ns] class which stores time information and allows us to perform some operations really fast. Lets start by firing up the required libraries: 

# COMMAND ----------

data = pd.read_csv('AirPassengers.csv')
data.head()

# COMMAND ----------

data.info()

# COMMAND ----------

# MAGIC %md
# MAGIC The data contains a particular month and number of passengers travelling in that month. But this is still not read as a TS object as the data types are ‘object’ and ‘int’. In order to read the data as a time series, we have to pass special arguments to the read_csv command:

# COMMAND ----------

#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html
data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m')
data.Month[0]

# COMMAND ----------

data.info()

# COMMAND ----------

data.set_index(['Month'], drop=True, inplace=True)
data.index = pd.DatetimeIndex(data.index.values, freq=data.index.inferred_freq)

data.head()

# COMMAND ----------

data.index

# COMMAND ----------

# MAGIC %md
# MAGIC #### Converting the dataframe into a series

# COMMAND ----------

ts = data['#Passengers']
ts.sort_index(inplace=True)
ts.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Using the index as a string constant:

# COMMAND ----------

ts['1949-01-01']

# COMMAND ----------

# MAGIC %md
# MAGIC Using the datetime library and use 'datetime' function:

# COMMAND ----------

ts[datetime(1949,1,1)]

# COMMAND ----------

# MAGIC %md
# MAGIC Specifying a time interval  

# COMMAND ----------

ts['1949-01-01':'1949-05-01']

# COMMAND ----------

ts[:'1949-10-01']

# COMMAND ----------

# MAGIC %md
# MAGIC There are 2 things to note here: 
# MAGIC + Unlike numeric indexing, the end index is included here.  
# MAGIC For instance, if we index a list as a[:5] then it would return the values at indices – [0,1,2,3,4].  
# MAGIC But here the index ‘1949-05-01’ was included in the output.  
# MAGIC + The indices have to be sorted for ranges to work. If you randomly shuffle the index, this won’t work.  

# COMMAND ----------

ts['1949']

# COMMAND ----------

# MAGIC %md
# MAGIC ### How to Check Stationarity of a Time Series?

# COMMAND ----------

# MAGIC %md
# MAGIC A TS is said to be stationary if its statistical properties such as mean, variance remain constant over time. But why is it important? Most of the TS models work on the assumption that the TS is stationary. Intuitively, we can sat that if a TS has a particular behaviour over time, there is a very high probability that it will follow the same in the future. Also, the theories related to stationary series are more mature and easier to implement as compared to non-stationary series.
# MAGIC 
# MAGIC Stationarity is defined using very strict criterion. However, for practical purposes we can assume the series to be stationary if it has constant statistical properties over time, ie. the following:
# MAGIC 
# MAGIC 1. constant mean  
# MAGIC 2. constant variance  
# MAGIC 3. an autocovariance that does not depend on time.  

# COMMAND ----------

# MAGIC %md
# MAGIC + The mean of the series should not be a function of time rather should be a constant. The first graph in image below has the left hand graph satisfying the condition whereas the graph in red has a time dependent mean.

# COMMAND ----------

# MAGIC %md
# MAGIC + The variance of the series should not a be a function of time. This property is known as homoscedasticity. The second graph below depicts what is and what is not a stationary series. (Notice the varying spread of distribution in the right hand graph)

# COMMAND ----------

# MAGIC %md
# MAGIC + The covariance of the i th term and the (i + m) th term should not be a function of time. In the third graph, you will notice the spread becomes closer as the time increases. Hence, the covariance is not constant with time for the ‘red series’.

# COMMAND ----------

display(Image('img/nonstationary.png'))

# COMMAND ----------

# MAGIC %md
# MAGIC Lets move onto the ways of testing stationarity. First and foremost is to simple plot the data and analyze visually. The data can be plotted using following command:

# COMMAND ----------

ts.plot(figsize=(12,8))

# COMMAND ----------

# MAGIC %md
# MAGIC It is clearly evident that there is an overall increasing trend in the data along with some seasonal variations. However, it might not always be possible to make such visual inferences (we’ll see such cases later). So, more formally, we can check stationarity using the following:
# MAGIC 
# MAGIC + Plotting Rolling Statistics: We can plot the moving average or moving variance and see if it varies with time. By moving average/variance I mean that at any instant ‘t’, we’ll take the average/variance of the last year, i.e. last 12 months. But again this is more of a visual technique.
# MAGIC 
# MAGIC + Dickey-Fuller Test: This is one of the statistical tests for checking stationarity. Here the null hypothesis is that the TS is non-stationary. The test results comprise of a Test Statistic and some Critical Values for difference confidence levels. If the ‘Test Statistic’ is less than the ‘Critical Value’, we can reject the null hypothesis and say that the series is stationary. Refer [this article](https://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/) for details.
# MAGIC 
# MAGIC These concepts might not sound very intuitive at this point. I recommend going through the prequel article. If you’re interested in some theoretical statistics, you can refer [Introduction to Time Series and Forecasting by Brockwell and Davis](https://www.amazon.com/Introduction-Forecasting-Springer-Texts-Statistics/dp/0387953515). The book is a bit stats-heavy, but if you have the skill to read-between-lines, you can understand the concepts and tangentially touch the statistics.
# MAGIC 
# MAGIC Back to checking stationarity, we’ll be using the rolling statistics plots along with Dickey-Fuller test results a lot so I have defined a function which takes a TS as input and generated them for us. Please note that I’ve plotted standard deviation instead of variance to keep the unit similar to mean.

# COMMAND ----------

#https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12,center=False).mean()
    rolstd = timeseries.rolling(window=12,center=False).std()

    #Plot rolling statistics:
    plt.figure(figsize=(12,8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

# COMMAND ----------

test_stationarity(ts)

# COMMAND ----------

# MAGIC %md
# MAGIC Though the variation in standard deviation is small, mean is clearly increasing with time and this is not a stationary series. Also, the test statistic is way more than the critical values. Note that the signed values should be compared and not the absolute values.
# MAGIC 
# MAGIC Next, we’ll discuss the techniques that can be used to take this TS towards stationarity.

# COMMAND ----------

# MAGIC %md
# MAGIC ### How to make a Time Series Stationary?
# MAGIC 
# MAGIC Though stationarity assumption is taken in many TS models, almost none of practical time series are stationary. So statisticians have figured out ways to make series stationary, which we’ll discuss now. Actually, its almost impossible to make a series perfectly stationary, but we try to take it as close as possible.
# MAGIC 
# MAGIC Lets understand what is making a TS non-stationary. There are 2 major reasons behind non-stationaruty of a TS:
# MAGIC + Trend – varying mean over time. In this case we saw that on average, the number of passengers was growing over time.
# MAGIC + Seasonality – variations at specific time-frames. eg people might have a tendency to buy cars in a particular month because of pay increment or festivals.
# MAGIC 
# MAGIC The underlying principle is to model or estimate the trend and seasonality in the series and remove those from the series to get a stationary series. Then statistical forecasting techniques can be implemented on this series. The final step would be to convert the forecasted values into the original scale by applying trend and seasonality constraints back.
# MAGIC 
# MAGIC Note: I’ll be discussing a number of methods. Some might work well in this case and others might not. But the idea is to get a hang of all the methods and not focus on just the problem at hand.
# MAGIC 
# MAGIC Let’s start by working on the trend part.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Estimating & Eliminating Trend
# MAGIC 
# MAGIC One of the first tricks to reduce trend can be transformation. For example, in this case we can clearly see that the there is a significant positive trend. So we can apply transformation which penalize higher values more than smaller values. These can be taking a log, square root, cube root, etc. Lets take a log transform here for simplicity:  

# COMMAND ----------

ts_log = np.log(ts)
ts_log.plot(figsize=(12,8))

# COMMAND ----------

# MAGIC %md
# MAGIC In this simpler case, it is easy to see a forward trend in the data. But its not very intuitive in presence of noise. So we can use some techniques to estimate or model this trend and then remove it from the series. There can be many ways of doing it and some of most commonly used are:
# MAGIC 
# MAGIC + Aggregation – taking average for a time period like monthly/weekly averages
# MAGIC + Smoothing – taking rolling averages
# MAGIC + Polynomial Fitting – fit a regression model
# MAGIC 
# MAGIC I will discuss smoothing here and you should try other techniques as well which might work out for other problems. Smoothing refers to taking rolling estimates, i.e. considering the past few instances. There are can be various ways but I will discuss two of those here.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Moving average
# MAGIC 
# MAGIC In this approach, we take average of ‘k’ consecutive values depending on the frequency of time series. Here we can take the average over the past 1 year, i.e. last 12 values. Pandas has specific functions defined for determining rolling statistics.

# COMMAND ----------

moving_avg = ts_log.rolling(window=12, center=False).mean()

fig, ax = plt.subplots(figsize=(12,8))
ts_log.plot(ax=ax)
moving_avg.plot(ax=ax)

# COMMAND ----------

# MAGIC %md
# MAGIC The orange line shows the rolling mean. Lets subtract this from the original series. Note that since we are taking average of last 12 values, rolling mean is not defined for first 11 values. This can be observed as:

# COMMAND ----------

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(15)

# COMMAND ----------

ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)

# COMMAND ----------

# MAGIC %md
# MAGIC This looks like a much better series. The rolling values appear to be varying slightly but there is no specific trend. Also, the test statistic is smaller than the 5% critical values so we can say with 95% confidence that this is a stationary series.
# MAGIC 
# MAGIC However, a drawback in this particular approach is that the time-period has to be strictly defined. In this case we can take yearly averages but in complex situations like forecasting a stock price, its difficult to come up with a number. So we take a ‘weighted moving average’ where more recent values are given a higher weight. There can be many technique for assigning weights. A popular one is exponentially weighted moving average where weights are assigned to all the previous values with a decay factor. Find details [here](http://pandas.pydata.org/pandas-docs/stable/computation.html#exponentially-weighted-moment-functions). This can be implemented in Pandas as:

# COMMAND ----------

expwighted_avg = ts_log.ewm(ignore_na=False, adjust=True, halflife=12, min_periods=0).mean()

fig, ax = plt.subplots(figsize=(12,8))
ts_log.plot(ax=ax)
expwighted_avg.plot(ax=ax)

# COMMAND ----------

# MAGIC %md
# MAGIC Note that here the parameter ‘halflife’ is used to define the amount of exponential decay. This is just an assumption here and would depend largely on the business domain. Now, let’s remove this from series and check stationarity:

# COMMAND ----------

ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)

# COMMAND ----------

# MAGIC %md
# MAGIC This TS has even lesser variations in mean and standard deviation in magnitude. Also, the test statistic is smaller than the 1% critical value, which is better than the previous case. Note that in this case there will be no missing values as all values from starting are given weights. So it’ll work even with no previous values.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Eliminating Trend and Seasonality
# MAGIC 
# MAGIC The simple trend reduction techniques discussed before don’t work in all cases, particularly the ones with high seasonality. Lets discuss two ways of removing trend and seasonality:
# MAGIC 
# MAGIC + Differencing – taking the difference with a particular time lag
# MAGIC + Decomposition – modeling both trend and seasonality and removing them from the model.

# COMMAND ----------

# MAGIC %md
# MAGIC Differencing
# MAGIC 
# MAGIC One of the most common methods of dealing with both trend and seasonality is differencing. In this technique, we take the difference of the observation at a particular instant with that at the previous instant. This mostly works well in improving stationarity.  
# MAGIC First order differencing can be done in Pandas as:

# COMMAND ----------

ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.plot(figsize=(12,8))

# COMMAND ----------

# MAGIC %md
# MAGIC This appears to have reduced trend considerably. Lets verify using our plots:

# COMMAND ----------

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that the mean and std variations have small variations with time. Also, the Dickey-Fuller test statistic is less than the 10% critical value, thus the TS is stationary with 90% confidence. We can also take second or third order differences which might get even better results in certain applications. I leave it to you to try them out.  

# COMMAND ----------

# MAGIC %md
# MAGIC Decomposing
# MAGIC 
# MAGIC In this approach, both trend and seasonality are modeled separately and the remaining part of the series is returned. I’ll skip the statistics and come to the results:

# COMMAND ----------

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

f, ax = plt.subplots(2,2,figsize=(12,12))
ax[0,0].plot(ts_log, label='Original')
ax[0,0].legend(loc='best')
ax[0,1].plot(trend, label='Trend')
ax[0,1].legend(loc='best')
ax[1,0].plot(seasonal,label='Seasonality')
ax[1,0].legend(loc='best')
ax[1,1].plot(residual, label='Residuals')
ax[1,1].legend(loc='best')
plt.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC Here we can see that the trend, seasonality are separated out from data and we can model the residuals. Lets check stationarity of residuals:  

# COMMAND ----------

ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)

# COMMAND ----------

# MAGIC %md
# MAGIC The Dickey-Fuller test statistic is significantly lower than the 1% critical value. So this TS is very close to stationary. You can try advanced decomposition techniques as well which can generate better results. Also, you should note that converting the residuals into original values for future data in not very intuitive in this case.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Forecasting a Time Series
# MAGIC 
# MAGIC We saw different techniques and all of them worked reasonably well for making the TS stationary. Lets make model on the TS after differencing as it is a very popular technique. Also, its relatively easier to add noise and seasonality back into predicted residuals in this case. Having performed the trend and seasonality estimation techniques, there can be two situations:
# MAGIC 
# MAGIC + A strictly stationary series with no dependence among the values. This is the easy case wherein we can model the residuals as white noise. But this is very rare.
# MAGIC 
# MAGIC + A series with significant dependence among values. In this case we need to use some statistical models like ARIMA to forecast the data.
# MAGIC 
# MAGIC Let me give you a brief introduction to ARIMA. I won’t go into the technical details but you should understand these concepts in detail if you wish to apply them more effectively. ARIMA stands for Auto-Regressive Integrated Moving Averages. The ARIMA forecasting for a stationary time series is nothing but a linear (like a linear regression) equation. The predictors depend on the parameters (p,d,q) of the ARIMA model:
# MAGIC 
# MAGIC + Number of AR (Auto-Regressive) terms (p): AR terms are just lags of dependent variable. For instance if p is 5, the predictors for x(t) will be x(t-1)….x(t-5).
# MAGIC + Number of MA (Moving Average) terms (q): MA terms are lagged forecast errors in prediction equation. For instance if q is 5, the predictors for x(t) will be e(t-1)….e(t-5) where e(i) is the difference between the moving average at ith instant and actual value.
# MAGIC + Number of Differences (d): These are the number of nonseasonal differences, i.e. in this case we took the first order difference. So either we can pass that variable and put d=0 or pass the original variable and put d=1. Both will generate same results.
# MAGIC 
# MAGIC An importance concern here is how to determine the value of ‘p’ and ‘q’. We use two plots to determine these numbers. Lets discuss them first.
# MAGIC 
# MAGIC + Autocorrelation Function (ACF): It is a measure of the correlation between the the TS with a lagged version of itself. For instance at lag 5, ACF would compare series at time instant ‘t1’…’t2’ with series at instant ‘t1-5’…’t2-5’ (t1-5 and t2 being end points).
# MAGIC 
# MAGIC + Partial Autocorrelation Function (PACF): This measures the correlation between the TS with a lagged version of itself but after eliminating the variations already explained by the intervening comparisons. Eg at lag 5, it will check the correlation but remove the effects already explained by lags 1 to 4.
# MAGIC 
# MAGIC The ACF and PACF plots for the TS after differencing can be plotted as:

# COMMAND ----------

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

# COMMAND ----------

lag_acf = acf(ts_log_diff, nlags=20, fft=False)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

# COMMAND ----------

#Plot ACF: 
f = plt.figure(figsize=(12,6))
ax1 = f.add_subplot(121)
ax2 = f.add_subplot(122)

ax1.plot(lag_acf)
ax1.axhline(y=0,linestyle='--',color='gray')
ax1.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
ax1.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
ax1.set_title('Autocorrelation Function')

#Plot PACF:
ax2.plot(lag_pacf)
ax2.axhline(y=0,linestyle='--',color='gray')
ax2.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
ax2.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
ax2.set_title('Partial Autocorrelation Function')
plt.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC In this plot, the two dotted lines on either sides of 0 are the confidence interevals. These can be used to determine the ‘p’ and ‘q’ values as:
# MAGIC 
# MAGIC + p – The lag value where the PACF chart crosses the upper confidence interval for the first time. If you notice closely, in this case p=2.
# MAGIC + q – The lag value where the ACF chart crosses the upper confidence interval for the first time. If you notice closely, in this case q=2.
# MAGIC 
# MAGIC Now, lets make 3 different ARIMA models considering individual as well as combined effects. I will also print the RSS for each. Please note that here RSS is for the values of residuals and not actual series.
# MAGIC 
# MAGIC We need to load the ARIMA model first:

# COMMAND ----------

import statsmodels.api as smapi

# COMMAND ----------

# MAGIC %md
# MAGIC The p,d,q values can be specified using the order argument of ARIMA which take a tuple (p,d,q). Let model the 3 cases:

# COMMAND ----------

# MAGIC %md
# MAGIC #### AR Model

# COMMAND ----------

model = smapi.tsa.arima.ARIMA(ts_log_diff, order=(2, 1, 0))  
results_AR = model.fit()  
plt.figure(figsize=(12,6))
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))

# COMMAND ----------

# MAGIC %md
# MAGIC #### MA Model

# COMMAND ----------

model = smapi.tsa.arima.ARIMA(ts_log_diff, order=(0, 1, 2))  
results_MA = model.fit()  
plt.figure(figsize=(12,6))
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Combined Model

# COMMAND ----------

model = smapi.tsa.arima.ARIMA(ts_log_diff, order=(2, 2, 2))  
results_ARIMA = model.fit(method_kwargs={"warn_convergence": False})  
plt.figure(figsize=(12,6))
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))

# COMMAND ----------

# MAGIC %md
# MAGIC Here we can see that the AR and MA models have almost the same RSS but combined is significantly better. Now, we are left with 1 last step, i.e. taking these values back to the original scale.
# MAGIC 
# MAGIC ### Taking it back to original scale
# MAGIC 
# MAGIC Since the combined model gave best result, lets scale it back to the original values and see how well it performs there. First step would be to store the predicted results as a separate series and observe it.

# COMMAND ----------

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that these start from ‘1949-02-01’ and not the first month. Why? This is because we took a lag by 1 and first element doesn’t have anything before it to subtract from. The way to convert the differencing to log scale is to add these differences consecutively to the base number. An easy way to do it is to first determine the cumulative sum at index and then add it to the base number. The cumulative sum can be found as:

# COMMAND ----------

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.head()

# COMMAND ----------

# MAGIC %md
# MAGIC You can quickly do some back of mind calculations using previous output to check if these are correct. Next we’ve to add them to base number. For this lets create a series with all values as base number and add the differences to it. This can be done as:

# COMMAND ----------

predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Here the first element is base number itself and from thereon the values cumulatively added. Last step is to take the exponent and compare with the original series.

# COMMAND ----------

predictions_ARIMA = np.exp(predictions_ARIMA_log)
predictions_ARIMA.head()

# COMMAND ----------

plt.figure(figsize=(10,6))
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))

# COMMAND ----------

# MAGIC %md
# MAGIC Finally we have a forecast at the original scale. Not a very good forecast I would say but you got the idea right? Now, I leave it upto you to refine the methodology further and make a better solution.
