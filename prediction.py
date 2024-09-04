#!/usr/bin/env python
# coding: utf-8

# # Task 2
# 
# ---
# 
# ## Predictive modeling of customer bookings
# 
# This Jupyter notebook includes some code to get you started with this predictive modeling task. We will use various packages for data manipulation, feature engineering and machine learning.
# 
# ### Exploratory data analysis
# 
# First, we must explore the data in order to better understand what we have and the statistical properties of the dataset.

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("data/customer_booking.csv", encoding="ISO-8859-1")
df.head()


# The `.head()` method allows us to view the first 5 rows in the dataset, this is useful for visual inspection of our columns

# In[3]:


df.info()


# The `.info()` method gives us a data description, telling us the names of the columns, their data types and how many null values we have. Fortunately, we have no null values. It looks like some of these columns should be converted into different data types, e.g. flight_day.
# 
# To provide more context, below is a more detailed data description, explaining exactly what each column means:
# 
# - `num_passengers` = number of passengers travelling
# - `sales_channel` = sales channel booking was made on
# - `trip_type` = trip Type (Round Trip, One Way, Circle Trip)
# - `purchase_lead` = number of days between travel date and booking date
# - `length_of_stay` = number of days spent at destination
# - `flight_hour` = hour of flight departure
# - `flight_day` = day of week of flight departure
# - `route` = origin -> destination flight route
# - `booking_origin` = country from where booking was made
# - `wants_extra_baggage` = if the customer wanted extra baggage in the booking
# - `wants_preferred_seat` = if the customer wanted a preferred seat in the booking
# - `wants_in_flight_meals` = if the customer wanted in-flight meals in the booking
# - `flight_duration` = total duration of flight (in hours)
# - `booking_complete` = flag indicating if the customer completed the booking
# 
# Before we compute any statistics on the data, lets do any necessary data conversion

# In[4]:


df["flight_day"].unique()


# In[5]:


mapping = {
    "Mon": 1,
    "Tue": 2,
    "Wed": 3,
    "Thu": 4,
    "Fri": 5,
    "Sat": 6,
    "Sun": 7,
}

df["flight_day"] = df["flight_day"].map(mapping)


# In[6]:


df["flight_day"].unique()


# In[7]:


df.describe()


# The `.describe()` method gives us a summary of descriptive statistics over the entire dataset (only works for numeric columns). This gives us a quick overview of a few things such as the mean, min, max and overall distribution of each column.
# 
# From this point, you should continue exploring the dataset with some visualisations and other metrics that you think may be useful. Then, you should prepare your dataset for predictive modelling. Finally, you should train your machine learning model, evaluate it with performance metrics and output visualisations for the contributing variables. All of this analysis should be summarised in your single slide.

# 
