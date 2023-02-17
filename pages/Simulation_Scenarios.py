#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import model2 as md


st.set_page_config(
     layout="wide",
     initial_sidebar_state="expanded"
 )
 
 
st.title(':green[Hospital Efficiency Project]')

st.header('Orthopaedic Planning Model')

"The simulation model will use the schedule created on the previous page and compare it with a default schedule in which 4 theatres operate 5 days per week, with 3 sessions per day.  Of those sessions, 2 will randomly allocate either 2 primary joint replacements or 1 revision joint replacement, while the third session will schedule 1 primary joint replacement."

"You can set up multiple scenarios to compare by changing the parameters in the sliders and clicking 'Add Scenario'."

	

with st.sidebar:
	st.markdown('# Model Parameters')
		
	st.markdown('## Ring-fenced beds:')
	n_beds = st.slider('Beds', 20, 60, md.DEFAULT_NUMBER_BEDS, 1)

	st.markdown('## Mean lengths-of-stay for each type of surgery:')
	primary_hip_los = st.slider('Primary Hip LoS', 1.0, 5.0, md.DEFAULT_PRIMARY_HIP_MEAN_LOS, 0.1)

	primary_knee_los = st.slider('Primary Knee LoS', 2.0, 5.0, md.DEFAULT_PRIMARY_KNEE_MEAN_LOS, 0.1)

	revision_hip_los = st.slider('Revision Hip LoS', 3.0, 8.0, md.DEFAULT_REVISION_HIP_MEAN_LOS, 0.1)

	revision_knee_los = st.slider('Revision Knee LoS', 3.0, 8.0, md.DEFAULT_REVISION_KNEE_MEAN_LOS, 0.1)

	unicompart_knee_los = st.slider('Unicompart knee LoS', 1.0, 4.0,md.DEFAULT_UNICOMPART_KNEE_MEAN_LOS, 0.1)
	    
	st.markdown('## Mean length of delayed discharge:')
	los_delay = st.slider('length of delay', 2.0, 10.0,md.DEFAULT_DELAY_POST_LOS_MEAN, 0.1)
		
	st.markdown('## Proportion of patients with a discharge delay:')
	prop_delay = st.slider('Proportion delayed', 0.00, 1.00,md.DEFAULT_PROB_WARD_DELAY, 0.01)
		
	st.markdown('## :green[Model execution]')
	replications = st.slider(':green[Multiple runs]', 1, 50, 10)
	runtime = st.slider(':green[Runtime (days)]', 30, 100, 60)

########################################################################
########### Add scenarios to session state to create scenarios dataframe
	
if 'counter' not in st.session_state:
	st.session_state['counter'] = 1

if 'scenarios_df' not in st.session_state:
	st.session_state['scenarios_df'] = pd.DataFrame({
			'Scenario':['Baseline'],
			'n_beds': [md.DEFAULT_NUMBER_BEDS],
			'Primary_hip_los': [md.DEFAULT_PRIMARY_HIP_MEAN_LOS],
			'Primary_knee_los': [md.DEFAULT_PRIMARY_KNEE_MEAN_LOS],
			'Revision_hip_los': [md.DEFAULT_REVISION_HIP_MEAN_LOS],
			'Revision_knee_los': [md.DEFAULT_REVISION_KNEE_MEAN_LOS],
			'Unicompart_knee_los': [md.DEFAULT_UNICOMPART_KNEE_MEAN_LOS],
			'Delay_mean': [md.DEFAULT_DELAY_POST_LOS_MEAN],
			'Delay_prop': [md.DEFAULT_PROB_WARD_DELAY],
			})

# Define function to add new scenario to session state

def add_scenario(n_beds, primary_hip_los, primary_knee_los, revision_hip_los, revision_knee_los, 
		unicompart_knee_los, los_delay, prop_delay):
	
    new_scenario = pd.DataFrame({
        'Scenario': [f'Scenario {st.session_state["counter"]}'],
        'n_beds': [n_beds],
        'Primary_hip_los': [primary_hip_los],
        'Primary_knee_los': [primary_knee_los],
        'Revision_hip_los': [revision_hip_los],
        'Revision_knee_los': [revision_knee_los],
        'Unicompart_knee_los': [unicompart_knee_los],
        'Delay_mean': [los_delay],
        'Delay_prop': [prop_delay]
    })
    st.session_state['scenarios_df'] = pd.concat([st.session_state['scenarios_df'], new_scenario], ignore_index=True)
    st.session_state['counter'] += 1

col1, col2, col3, col4 = st.columns(4)
with col1:
	if st.button('Add Scenario'):
		add_scenario(n_beds, primary_hip_los, primary_knee_los, revision_hip_los, 
		revision_knee_los, unicompart_knee_los, los_delay, prop_delay)
with col2:
	if st.button('Remove Scenario'):
		st.session_state['scenarios_df'] = st.session_state['scenarios_df'].iloc[:-1]

st.dataframe(st.session_state['scenarios_df'])

########### add new schedule

if st.button('Use newly defined theatre schedule in your simulation'):
	st.write("Checking if you have created a theatre schedule...")
	# Retrieve the DataFrame from SessionState:
	try:
		schedule_scenario = st.session_state.schedule_scenario
		"Your schedule looks like this:"
		st.write(st.session_state['schedule_scenario'].head())
		"All your chosen scenarios will be run with both the baseline and the newly created schedule."
		
	except:
		st.error("No schedule has been created: go to Flexible Schedule tab and generate a schedule if you want to use a schedule other than the default")
		st.session_state.schedule_scenario = None

############ set up scenarios

schedule = md.Schedule()
args = md.Scenario(schedule)


def get_scenario_dict(df):
	'''
	Creates a dictionary object of scenario attributes
	    
	Returns:
	--------
	dict
	Contains the scenario attributes for the model
	'''
		    
	scenario_dict = {}
	for i, row in df.iterrows():
	    key = row['Scenario']
	    value = [{'n_beds': row['n_beds']},
		     {'primary_hip_mean_los': row['Primary_hip_los']},
		     {'primary_knee_mean_los': row['Primary_knee_los']},
		     {'revision_hip_mean_los': row['Revision_hip_los']},
		     {'revision_knee_mean_los': row['Revision_knee_los']},
		     {'unicompart_knee_mean_los' : row['Unicompart_knee_los']},
		     {'delay_post_los_mean': row['Delay_mean']},
		     {'prob_ward_delay': row['Delay_prop']}]
	    scenario_dict[key] = value
	    
	return(scenario_dict)
	        
    
def get_scenarios(dict, new_schedule):

	"""
	Create dictionary of scenario objects using attribute dictionary
	A set of baseline schedule
	A set of new_schedule
	
	Returns:
	--------
	dict
	Contains the scenarios for the model
  
	"""
	dict = get_scenario_dict(st.session_state['scenarios_df'])
	scenarios = {}
	
	for key, value in dict.items():
		attributes = {}
		for item in value:
			for sub_key, sub_value in item.items():
				attributes[sub_key] = sub_value
		scenarios[key] = md.Scenario(schedule,**attributes)
		
		if 'schedule_scenario' in st.session_state:
			st.write("New schedule is in session state and will be added to scenarios")
			new_schedule=st.session_state['schedule_scenario']
			st.write(new_schedule.head())
			# Create a scenario object with new schedule
			scenarios[f'{key}_new_schedule'] = md.Scenario(schedule, schedule_avail = new_schedule, **attributes)
    		   
	return scenarios
 

    
####### Run simulations    
    
if st.button('Start simulation', type='primary'):
    
	with st.spinner('Simulating...'):

		#get the scenarios
		if 'schedule_scenario' in st.session_state:
			st.write("New schedule is in session state and will be simulated")
			scenarios = get_scenarios(dict, st.session_state['schedule_scenario'])
		else:
			st.write("New sched not found and won't be simulated")
			scenarios = get_scenarios(dict, None)

		#run the scenario analysis for all results
		scenario_results = md.run_scenario_analysis(scenarios, 
			                                 runtime+md.DEFAULT_WARM_UP_PERIOD,
			                                 n_reps= replications)
		#create single patient-level frame
		scenario_results_patients = {key: pd.concat([scenario_results[2][key], scenario_results[3][key]], 
			                      ignore_index=True) for key in scenario_results[2].keys()}
	  

		st.success('Done!')
		
	col1, col2 = st.columns(2)
	with col1:
		st.write("Summary of overall throughput across the model runtime:")
		patient_summary = md.patient_scenarios(scenario_results_patients)
		table = md.total_thruput_table(scenario_results_patients)
		st.write(table)
		st.write("The plots represent mean outputs for each scenario")
		
		"Bed utilisation across the model runtime helps to understand how different scenarios affect the utilisation of beds each day and week."
		
		"This is summarised for all weeks to show average bed utilisation for each day of week, for each of the scenarios investigated."
		
		"'Lost slots' represents a mismatch between the number of patients scheduled for surgery, and the number of beds available to them, given patient lengths-of-stay."
		"Some of these lost slots may be accounted for by theatre cancellations for patient reasons.  Some may involve some bed management. "
		"Others will result in lost theatre slots, if a bed isn't available for the patient."
		
		"Total throughput represents the average number of surgeries that can be performed per week, for each scenario investigated."
		st.pyplot(md.total_thruput_pt_results(scenario_results_patients))
	with col2:
		" "
		" "
		" "
		st.pyplot(md.scenario_daily_audit(scenario_results[1]))
		st.pyplot(md.scenario_weekly_audit(scenario_results[1]))
		st.pyplot(md.lost_slots(patient_summary))


