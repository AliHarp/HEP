#!/usr/bin/env python
# coding: utf-8

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
import streamlit as st
from PIL import Image
import model2 as md

#st.cache_data.clear()

st.set_page_config(
     layout="wide",
     initial_sidebar_state="expanded"
 )

#DEFAULT VALUES FROM SIMULATION
###########################################################################

warm_up_period = md.DEFAULT_WARM_UP_PERIOD
results_collection_period = md.DEFAULT_RESULTS_COLLECTION_PERIOD

schedule_avail = pd.DataFrame()

weekday = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sessions_per_weekday = {'Monday': 3, 'Tuesday': 3, 'Wednesday': 3, 'Thursday': 3, 'Friday': 3, 'Saturday': 3, 'Sunday': 3}
allocation = {'Monday': ['2P_or_1R', '2P_or_1R', '1P'], 'Tuesday': ['2P_or_1R', '2P_or_1R','1P'], 'Wednesday': ['2P_or_1R', '2P_or_1R','1P'], 
              'Thursday': ['2P_or_1R', '2P_or_1R', '1P'], 'Friday': ['2P_or_1R', '2P_or_1R', '1P'], 'Saturday': ['2P_or_1R', '2P_or_1R', '1P'], 'Sunday': ['2P_or_1R', '2P_or_1R', '1P']}
theatres_per_weekday = {'Monday': 4, 'Tuesday': 4, 'Wednesday': 4, 'Thursday': 4, 'Friday': 4, 'Saturday': 4, 'Sunday': 4}
sessions_per_weekday_list = list(sessions_per_weekday.values())

st.title(':green[Hospital Efficiency Project]')

st.header('Orthopaedic Planning Model: Theatre Schedule')

st.sidebar.title("Weekly Schedule")
st.sidebar.markdown(":orange[For each weekday, please select:]")
st.sidebar.markdown(":red[(i)]   *:red[the number of sessions per day;]*")
st.sidebar.markdown(":red[(ii)]  *:red[the surgical categories for each session]*")
st.sidebar.markdown(":red[(iii)] *:red[the number of theatres.]*")


##########################################################################################
#####################  FUNCTIONS TO GENERATE SCHEDULE  ###############################
##########################################################################################

#### CREATE DICTIONARY WITH WEEKDAY KEYS AND VALUES ARE LISTS OF LISTS OF ALLOCATIONS PER SESSION PER THEATRE


def create_schedule(weekday, sessions_per_weekday_list, allocation, theatres_per_weekday):
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


### RETURN A ONE-WEEK SCHEDULE DATAFRAME

def daily_counts(day_data):
	"""
	day_data: called in week_schedule() function, day_data is a sample weekly dictionary from create_schedule()
	Returns two lists for one week of primary and revision counts
	"""
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
	week_sched = pd.DataFrame(columns=['Primary_slots', 'Revision_slots'])
	day_data = create_schedule(weekday, sessions_per_weekday_list, allocation, theatres_per_weekday)
	for key, value in day_data.items():
		week_sched.loc[key] = daily_counts(value)
	week_sched = week_sched.reset_index()
	week_sched.rename(columns = {'index':'Day'}, inplace = True)
	return week_sched



#### USE PREVIOUS TWO FUNCTIONS TO GENERATE SCHEDULE FOR RUN LENGTH + WARM_UP
#@st.cache
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
        

############################################################################
###############   USER INPUTS TO UPDATE SCHEDULE VALUES   ############################################
############################################################################


# INPUTS FOR CREATING SCHEDULE BY DAY OF WEEK, SESSIONS PER DAY, SURGICAL ALLOCATIONS PER SESSION

for day,session,session_key, allocate, allocate_key, theatre, theatre_key, keylist, indices in zip(weekday, sessions_per_weekday.values(), 
	sessions_per_weekday.keys(),allocation.values(), allocation.keys(), theatres_per_weekday.values(), theatres_per_weekday.keys(),
	[i for i in range(1,8)], [i for i in enumerate(weekday)]): 
	
	####################################################################
	#select weekday - default values provided as indices of weekday list
	
	selected_weekday = day
	#selected_weekday = st.selectbox("Select weekday:", weekday, key=keylist, index=indices[0])
	st.write(":red[Plan your sessions and number of theatres for]", selected_weekday)
	
	with st.expander(f"View scheduling details for {day}"):
	
		col1, col2 = st.columns(2)
		
		with col1:
		#select number of sessions per weekday
			st.write(f":green[**1.**	Please select the number of theatre sessions required for {day}]")
			selected_sessions = st.slider("Number of sessions:", 0, 4, 3, key=keylist*10)
			st.write("Selected sessions:", selected_sessions)
			sessions_per_weekday[session_key] = selected_sessions
			sessions_per_weekday_list = list(sessions_per_weekday.values())
			
		with col2:
			st.write(":green[**2.**	Please select the surgery allocation for each session.]")
			st.markdown(f"You may allocate {selected_sessions} of the following surgical categories:")
			st.markdown(" -  **1R:** *1 revision*")
			st.markdown(" -  **2P:** *2 primary*")
			st.markdown(" -  **1R or 2P:** *random allocation of 1 revision or 2 primary*")
			st.markdown(" -  **1P:** *1 primary*")
		
			#######################################################################
			#select theatre allocations per session - remove default selections if fewer sessions selected than default
			
			if selected_sessions >= len(allocate):
				selected_allocations = st.multiselect("Allocations:", ['1R','1P','2P','2P_or_1R','1R','1P','2P','2P_or_1R'], 
				default=allocate, max_selections = selected_sessions, key=keylist*100)
			else:
				selected_allocations = st.multiselect("Allocations:", ['1R','1P','2P','2P_or_1R','1R','1P','2P','2P_or_1R'], 
				default=None, max_selections = selected_sessions, key=keylist*100)	
			
			# clarify options of 0 sessions selected		
			if selected_sessions == 0:
				selected_allocations = []
				st.write(":orange[You have set the session number to 0, no selection available]")
			
			# count down remaining sessions to be allocated	
			remaining = selected_sessions - len(selected_allocations)
			if (len(selected_allocations) < selected_sessions):
				st.write(f":orange[**You have {remaining} options remaining - all sessions must be allocated**]")
				
			st.write("Selected allocations:", selected_allocations)
			allocation[allocate_key] = selected_allocations
		
		#########################################################################
		#select number of theatres - if number of sessions is set to 0, number of theatre will be set to 0

		with col1:
			st.write(f":green[**3.**	Please select the number of operating theatres required for {day}]")
			
			# if no sessions are chosen, number of theatres must also be 0
			if selected_sessions == 0:
				selected_theatres = 0
				selected_theatres = st.slider("Number of theatres:", 0, 6, 0, key=keylist*1000)
			else:
				selected_theatres = st.slider("Number of theatres:", 0, 6, 4, key=keylist*1000)
				
			st.write(":orange[If you have selected 0 sessions per day, the number of theatres in use will also be 0]")
			st.write("Selected theatres:", selected_theatres)
			theatres_per_weekday[theatre_key] = selected_theatres
#@st.cache
def new_sched_rules_df():
	"""
	params: 
	weekday is list of weekdays
	sessions_per_weekday_list is number of sessions per day
	allocations per session
	theatres per session
	
	returns: a df of theatre rules for the new user schedule
	"""
	new_sched_rules = pd.DataFrame(list(zip(weekday, sessions_per_weekday_list, allocation.values(), theatres_per_weekday.values())), 
		columns =['Weekday', 'Sessions', 'Allocations', 'Theatre numbers'])
	return new_sched_rules	      
		       		

#Refresh schedule with new values if no conflict between number of sessions and allocations.  Error message if conflict.		
if st.button('Generate schedule', type='primary'):
	alln = True
	while alln == True:
		if all(session == len(allocate) for session, allocate, allocate_keys in zip(sessions_per_weekday_list, allocation.values(), allocation.keys())):
			schedule_scenario = create_full_schedule()
			st.write("Done!")
			st.write("You can view a sample of the theatre schedule in :orange['sample two-weekly schedule'] tab")
			if 'schedule_scenario' not in st.session_state:
				st.session_state.schedule_scenario = schedule_scenario
			alln = False
			break
		else:
			if (session > len(allocate) for session, allocate, allocate_keys in zip(sessions_per_weekday_list, allocation.values(), allocation.keys())):
				st.write(f":green[**Please check that you have allocated all sessions**]") 
				st.write(":green[The highlighted rows in the weekday scheduling table show days with incorrectly allocated sessions]")
				break
	new_sched_rules = new_sched_rules_df()
	if 'new_sched_rules' not in st.session_state:
		st.session_state.new_sched_rules = new_sched_rules	       		
			
	
tab1, tab2 = st.tabs(["Weekday scheduling values","A sample two-weekly schedule"])


# Display table of selected theatre options.  Highlight row/s with conflict between number sessions and sessions allocated.  			
with tab1:
	df = pd.DataFrame(list(zip(weekday, sessions_per_weekday_list, allocation.values(), theatres_per_weekday.values())), 
		columns =['Weekday', 'Sessions', 'Allocations', 'Theatre numbers'])
     
	def highlight(s):
		is_highlight = (s[1]) > (len(s[2]))
		return ['background-color: yellow' if is_highlight else '' for i in range(len(s))]

	st.dataframe(df.style.apply(highlight, axis=1))


# Display sample 2 week schedule where number sessions == sessions allocated.
# Error message if no schedule created.  Clear session state if change has been made to schedule and not regenerated. 
# This is so the simulation doesn't use the previous version. 	
with tab2:
	try:
		st.dataframe(schedule_scenario.head(14))

	except:
		st.error("No schedule has been created")
		try:
			del st.session_state['schedule_scenario']
			del st.session_state['new_sched_rules']
		except: 
			st.stop()
		st.stop()
     
	




