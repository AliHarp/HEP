#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objs as go
import model2 as md


st.set_page_config(
     layout="wide",
     initial_sidebar_state="expanded"
 )
 
 
st.title(':green[Hospital Efficiency Project]')

st.header('Orthopaedic Planning Model: Simulation')

"The simulation model will use the schedule created on the previous page and compare it with a default schedule in which 4 theatres operate 5 days per week, with 3 sessions per day.  Of those sessions, 2 will randomly allocate either 2 primary joint replacements or 1 revision joint replacement, while the third session will schedule 1 primary joint replacement."

"You can set up multiple scenarios for comparison by changing the parameters in the sliders and clicking 'Add Scenario'."

schedule = md.Schedule()
args = md.Scenario(schedule)

# default slider values from model2
default_values = {'n_beds': md.DEFAULT_NUMBER_BEDS, 
		'primary_hip_los': md.DEFAULT_PRIMARY_HIP_MEAN_LOS,
		'primary_knee_los': md.DEFAULT_PRIMARY_KNEE_MEAN_LOS,
		'revision_hip_los': md.DEFAULT_REVISION_HIP_MEAN_LOS,
		'revision_knee_los': md.DEFAULT_REVISION_KNEE_MEAN_LOS,
		'unicompart_knee_los': md.DEFAULT_UNICOMPART_KNEE_MEAN_LOS,
		'los_delay': md.DEFAULT_DELAY_POST_LOS_MEAN,
		'prop_delay': md.DEFAULT_PROB_WARD_DELAY}

#used to display the default scheduling rules used by model2, for comparison with the user-defined 'new' schedule		
default_sched_rules = pd.DataFrame(list(zip(md.SET_WEEKDAY, md.SET_SESSIONS_PER_WEEKDAY_LIST, md.SET_ALLOCATION.values(), md.SET_THEATRES_PER_WEEKDAY.values())),
		       columns =['Weekday', 'Sessions', 'Allocations', 'Theatre numbers'])

#initialise counter into session state
if 'counter' not in st.session_state:
	st.session_state['counter'] = 1
key = 'slider_'
slider_key = f'{key}{st.session_state["counter"]}'

#set up sliders using default values from model2
with st.sidebar:
	st.markdown('# Model Parameters')
		
	st.markdown('## Ring-fenced beds:')
	n_beds = st.slider('Beds', 20, 60, default_values['n_beds'], 1)

	st.markdown('## Mean lengths-of-stay for each type of surgery:')
	primary_hip_los = st.slider('Primary Hip LoS', 1.0, 5.0, default_values['primary_hip_los'], 0.1, key=slider_key)

	primary_knee_los = st.slider('Primary Knee LoS', 1.0, 5.0, default_values['primary_knee_los'], 0.1)

	revision_hip_los = st.slider('Revision Hip LoS', 2.0, 8.0, default_values['revision_hip_los'], 0.1)

	revision_knee_los = st.slider('Revision Knee LoS', 2.0, 8.0, default_values['revision_knee_los'], 0.1)

	unicompart_knee_los = st.slider('Unicompart knee LoS', 1.0, 4.0,default_values['unicompart_knee_los'], 0.1)
	    
	st.markdown('## Mean length of delayed discharge:')
	los_delay = st.slider('length of delay', 2.0, 10.0,default_values['los_delay'], 0.1)
		
	st.markdown('## Proportion of patients with a discharge delay:')
	prop_delay = st.slider('Proportion delayed', 0.00, 1.00, default_values['prop_delay'], 0.01)
		
	st.markdown('## :green[Model execution]')
	replications = st.slider(':green[Multiple runs]', 1, 50, 10)
	runtime = st.slider(':green[Runtime (days)]', 30, 100, 60)

md.DEFAULT_RESULTS_COLLECTION_PERIOD = runtime	
########################################################################
########### Initialise baseline scenarios to session state to create baseline scenarios dataframe


if 'scenarios_df' not in st.session_state:
	st.session_state['scenarios_df'] = pd.DataFrame({
			'Ward Configuration':['Baseline'],
			'Number of beds': [md.DEFAULT_NUMBER_BEDS],
			'Primary hip LoS': [md.DEFAULT_PRIMARY_HIP_MEAN_LOS],
			'Primary knee LoS': [md.DEFAULT_PRIMARY_KNEE_MEAN_LOS],
			'Revision hip LoS': [md.DEFAULT_REVISION_HIP_MEAN_LOS],
			'Revision knee LoS': [md.DEFAULT_REVISION_KNEE_MEAN_LOS],
			'Unicompart knee LoS': [md.DEFAULT_UNICOMPART_KNEE_MEAN_LOS],
			'Mean delay LoS': [md.DEFAULT_DELAY_POST_LOS_MEAN],
			'Proportion delayed': [md.DEFAULT_PROB_WARD_DELAY]
			})

# Define function to add new scenario to session state baseline scenarios dataframe

def add_scenario(n_beds, primary_hip_los, primary_knee_los, revision_hip_los, revision_knee_los, 
		unicompart_knee_los, los_delay, prop_delay):
	
    new_scenario = pd.DataFrame({
        'Ward Configuration': [f'Scenario {st.session_state["counter"]}'],
        'Number of beds': [n_beds],
        'Primary hip LoS': [primary_hip_los],
        'Primary knee LoS': [primary_knee_los],
        'Revision hip LoS': [revision_hip_los],
        'Revision knee LoS': [revision_knee_los],
        'Unicompart knee LoS': [unicompart_knee_los],
        'Mean delay LoS': [los_delay],
        'Proportion delayed': [prop_delay]
    })
    st.session_state['scenarios_df'] = pd.concat([st.session_state['scenarios_df'], new_scenario], ignore_index=True)
    st.session_state['counter'] += 1
    
    
# buttons to add and remove scenarios from scenario dataframe
col1, col2, col3, col4 = st.columns([1,1,2,2])
with col1:
	if st.button('Add Scenario'):
		add_scenario(n_beds, primary_hip_los, primary_knee_los, revision_hip_los, 
		revision_knee_los, unicompart_knee_los, los_delay, prop_delay)
with col2:
	if st.button('Remove Scenario'):
		# Create a copy of the dataframe without the last row to ensure baseline scenario can't be removed!
		new_df = st.session_state['scenarios_df'].iloc[:-1].copy()
		if st.session_state['counter'] >= 2:
			st.session_state['counter'] -= 1
		# If the dataframe has only one row, do not remove it
		if len(new_df) > 0:
    			# Update the session state with the new dataframe
    			st.session_state['scenarios_df'] = new_df

# display scenarios dataframe
st.dataframe(st.session_state['scenarios_df'])

st.markdown("***")

########## display default schedule plus new sched if generated

# sample of default sched
default_sched = md.schedule.theatre_capacity()

########### add new schedule to scenarios if required

if st.button('Use newly defined theatre schedule in your simulation', type='primary'):
	st.markdown("***")
	st.write(":green[Checking if you have created a theatre schedule...]")
	st.markdown("***")
	# Retrieve the DataFrame from SessionState:
	try:
		# define new schedule scenario using new scheduled generated in Flexible_Theatre_Schedule
		schedule_scenario = st.session_state.schedule_scenario
		new_sched_rules = st.session_state.new_sched_rules
		#st.write(st.session_state['new_sched_rules'])
		#st.write("sched in after setting")
		#st.write(st.session_state['schedule_scenario'].head(7))
		col1, col2 = st.columns(2)
		if 'schedule_scenario' in st.session_state: 
			with col1:
				":orange[These are your new theatre schedule rules:]"
				new_sched_rules = st.session_state.new_sched_rules
				st.write(st.session_state['new_sched_rules'])
				#st.write(st.session_state['schedule_scenario'].head(7))
			with col2:
				":orange[These are the default theatre schedule rules:]"
				default_sched_rules = default_sched_rules
				st.dataframe(default_sched_rules)
				#st.write(default_sched.head(7))
			st.markdown("***")
			":green[**All your chosen scenarios will run with both the baseline and the newly created schedule.**]"
			st.markdown("***")
			new_sched_df = st.session_state['scenarios_df'].copy()
			new_sched_df = pd.concat([new_sched_df, new_sched_df], ignore_index=True)
			new_sched_df['Theatre Schedule'] = np.repeat(['DEFAULT', 'NEW'], len(st.session_state['scenarios_df']))
			st.dataframe(new_sched_df)
	except:
		st.error("No schedule has been created: go to Flexible Theatre Schedule tab and generate a schedule if you want to use a schedule other than the default")
		":orange[The default schedule looks like this:]"
		st.dataframe(default_sched_rules)
		#if 'schedule_scenario' in st.session_state: 
		#	del st.session_state['schedule_scenario']



############ set up scenarios

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
	    key = row['Ward Configuration']
	    value = [{'n_beds': row['Number of beds']},
		     {'primary_hip_mean_los': row['Primary hip LoS']},
		     {'primary_knee_mean_los': row['Primary knee LoS']},
		     {'revision_hip_mean_los': row['Revision hip LoS']},
		     {'revision_knee_mean_los': row['Revision knee LoS']},
		     {'unicompart_knee_mean_los' : row['Unicompart knee LoS']},
		     {'delay_post_los_mean': row['Mean delay LoS']},
		     {'prob_ward_delay': row['Proportion delayed']}]
	    scenario_dict[key] = value
	    
	return(scenario_dict)
	        
    
def get_scenarios(dict_s, new_schedule):

	"""
	Create dictionary of scenario objects using attribute dictionary
	A set of baseline schedule
	A set of new_schedule if required
	
	Returns:
	--------
	dict
	Contains the scenarios for the model
  
	"""
	scenarios = {}
	
	for key, value in dict_s.items():
		attributes = {}
		for item in value:
			for sub_key, sub_value in item.items():
				attributes[sub_key] = sub_value
		#for each scenario, create a Scenario object with params as kwargs 		
		scenarios[key] = md.Scenario(schedule, **attributes)
		
		# same kwargs for each scenario but change schedule_avail into new_schedule 		
		if 'schedule_scenario' in st.session_state:
			new_schedule=st.session_state['schedule_scenario']
			# Create a scenario object with new schedule so all scenarios run with both
			scenarios[f'{key}_new_schedule'] = md.Scenario(schedule, schedule_avail = new_schedule, **attributes)
	    		   
	return scenarios
 

    
####### Run simulations 
    
if st.button('Start simulation', type='primary'):
	with st.spinner('Simulating...'):
		dict_s = get_scenario_dict(st.session_state['scenarios_df'])
		#get the scenarios
		if 'schedule_scenario' in st.session_state:
			scenarios = get_scenarios(dict_s, st.session_state['schedule_scenario'])
			#st.write("A new schedule is also being used!")
		else:
		#get the scenarios if no new schedule is being used
			st.write("A new schedule hasn't been found so only the default schedule will be used")
			scenarios = get_scenarios(dict_s, None)

		#run the scenario analysis for all results
		scenario_results = md.run_scenario_analysis(scenarios, 
					                    runtime+md.DEFAULT_WARM_UP_PERIOD,
					                    n_reps= replications)
		#create single patient-level frame
		scenario_results_patients = {key: pd.concat([scenario_results[2][key], scenario_results[3][key]], 
					              ignore_index=True) for key in scenario_results[2].keys()}
		  

		st.success('Done!')
		
	st.write("Summary of overall throughput across the model runtime:")
	patient_summary = md.patient_scenarios(scenario_results_patients)
	table = md.total_thruput_table(scenario_results_patients)
	st.write(table)	
	
	# display results 
	col1, col2 = st.columns([2,1])
	with col2:

		st.write("The plots represent mean results for each scenario")
		
		":orange[Bed utilisation] across the model runtime helps to understand how different scenarios affect the utilisation of beds each day and week."
		
		"This is summarised for all weeks to show average bed utilisation for each day of week, for each of the scenarios investigated."
		
		":orange['Lost slots'] represents a mismatch between the number of patients scheduled for surgery, and the number of beds available to them, given patient lengths-of-stay."
		"Some of these lost slots may be accounted for by theatre cancellations for patient reasons.  Some may involve some bed management. "
		"Others will result in lost theatre slots, if a bed isn't available for the patient."
		"Lost slots therefore give a good indication of scenarios which are putting the system under pressure."
		
		":orange[Total throughput] represents the average number of surgeries that can be performed per week, for each scenario investigated."
			

		
	with col1:

		if 'schedule_scenario' in st.session_state or len(st.session_state['scenarios_df'])>1:
			st.pyplot(md.scenario_daily_audit(scenario_results[1]))
			st.pyplot(md.scenario_weekly_audit(scenario_results[1]))
			st.pyplot(md.lost_slots(patient_summary))
			st.pyplot(md.total_thruput_pt_results(scenario_results_patients))
		else:
			    # Get results for single scenario only by running multiple_reps
			results = md.multiple_reps(args, n_reps = replications, results_collection=runtime+md.DEFAULT_WARM_UP_PERIOD)
			m_results = results[0]
			m_day_results = results[1]
			m_primary_pt_results = results[2]
			m_revision_pt_results = results[3]
 	
			st.pyplot(md.weekly_summ_bed_utilisation(m_day_results))
			st.pyplot(md.daily_summ_bed_utilisation(m_day_results))
			

# clear simulation and schedule 
col1, col2 = st.columns([2,1])
with col2:
	if st.button('Reset simulation', type='primary'):
		st.session_state['scenarios_df'] = st.session_state['scenarios_df'].iloc[[0]]
		st.session_state['counter'] = 1
		if 'schedule_scenario' in st.session_state:
			del st.session_state['schedule_scenario']   
