#!/usr/bin/env python
# coding: utf-8

# ## Flexible Theatre Scheduling
# 
# 
# 1. Choose number of sessions per weekday
# 2. Choose surgeries per session: (1P) 1 primary; (1R) 1 revision (2P) 2 primaries (2P\1R) random allocation of 2 primary or 1 revision
# 3. Number of theatres per weekday
# 
# NOTE: This means each theatre per weekday will have the same number of sessions and the same surgery allocation 


import simpy
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import itertools
import arrow
import random
import math
import warnings



#### NEW PARAMETERS NEEDED FOR WEEKLY SCHEDULE

warm_up_period = 14
results_collection_period = 40

weekday = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sessions_per_weekday_dict = {'Monday': 3, 'Tuesday': 3, 'Wednesday': 3, 'Thursday': 3, 'Friday': 3, 'Saturday': 1, 'Sunday': 0}
sessions_per_weekday_list = list(sessions_per_weekday_dict.values())
allocation = {'Monday': ['2P_or_1R', '2P_or_1R', '1P'], 'Tuesday': ['2P_or_1R', '2P_or_1R','1P'], 'Wednesday': ['2P_or_1R', '2P_or_1R','1P'], 
              'Thursday': ['2P_or_1R', '2P_or_1R', '1P'], 'Friday': ['2P_or_1R', '2P_or_1R', '1P'], 'Saturday': ['2P_or_1R'], 'Sunday': []}
theatres_per_weekday = {'Monday': 4, 'Tuesday': 4, 'Wednesday': 4, 'Thursday': 4, 'Friday': 4, 'Saturday': 2, 'Sunday': 0}


#### CREATE DICTIONARY WITH WEEKDAY KEYS AND VALUES ARE LISTS OF LISTS OF ALLOCATIONS PER SESSION PER THEATRE

def create_schedule(weekday, sessions_per_weekday, allocation, theatres_per_weekday):
    """
    Arguments needed:
        *weekday: a list of weekdays
        *sessions_per_weekday: a list of integers representing the number of sessions per weekday
        *allocation: a dictionary where the keys are the weekdays and the values are lists of 
                    allocations for each session 
        *theatres_per_weekday: a dictionary where the keys are the weekdays and the values are 
                    integers representing the number of theatres per weekday 
    Returns a dictionary where the keys are the weekdays and the values are lists 
                    of lists of allocations for each theatre for each session.
    """
    schedule = {}
    for day, num_sessions in zip(weekday, sessions_per_weekday_list):
        schedule[day] = []
        for theatre in range(theatres_per_weekday[day]):
            schedule[day].append([])
            for session in range(num_sessions):
                if allocation[day][session] == '1P':
                    schedule[day][theatre].append({'primary': 1})
                elif allocation[day][session] == '1R':
                    schedule[day][theatre].append({'revision': 1})
                elif allocation[day][session] == '2P':
                    schedule[day][theatre].append({'primary': 2})
                elif allocation[day][session] == '2P_or_1R':
                    if random.random() > 0.5:
                        schedule[day][theatre].append({'primary': 2})
                    else:
                        schedule[day][theatre].append({'revision': 1})
    return schedule


# RETURN A ONE-WEEK SCHEDULE DATAFRAME

def daily_counts(day_data):
    """
    day_data: called in week_schedule() function, day_data is a sample weekly dictionary from create_schedule()
    Returns two lists for one week of primary and revision counts
    """
    #day_data = create_schedule(weekday, sessions_per_weekday, allocation, theatres_per_weekday)
    primary_slots = 0
    revision_slots = 0
    for value in day_data:
        if value:
            for sub_value in value:
                if 'primary' in sub_value:
                    primary_slots += sub_value['primary']
                if 'revision' in sub_value:
                    revision_slots += sub_value['revision']
    return [primary_slots, revision_slots]

def week_schedule():
    """
    samples a weekly dictionary of theatres, sessions, and surgeries from create_schedule()
    Convert daily_count() counts to a pandas DataFrame with 'primary' and 'revision' datily counts as columns 
    with weekdays and day index.
    """
    week_sched = pd.DataFrame(columns=['Primary_slots', 'revision_slots'])
    day_data = create_schedule(weekday, sessions_per_weekday_list, allocation, theatres_per_weekday)
    for key, value in day_data.items():
        week_sched.loc[key] = daily_counts(value)
    week_sched = week_sched.reset_index()
    week_sched.rename(columns = {'index':'Day'}, inplace = True)
    return week_sched


#### USE PREVIOUS TWO FUNCTIONS TO GENERATE SCHEDULE FOR RUN LENGTH + WARM_UP

def create_full_schedule():
    """  
    Determine length of schedule
    Run week_schedule for length of full runtime + warmup
    Return dataframe of full schedule for simulation to read by day
    """
    length_sched = int(round(2*(warm_up_period+results_collection_period)/7, 0))

    schedule_avail = pd.DataFrame()
    for week in range(length_sched):
        single_random_week = week_schedule()
        schedule_avail = pd.concat([schedule_avail, single_random_week],axis=0)
    return schedule_avail.reset_index()
        

schedule_avail = create_full_schedule()
print(schedule_avail.head())






