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

"You can change the parameters of the simulation model and re-run it."  

"It uses a default schedule in which 4 theatres operate 5 days per week, with 3 sessions per day. Of those sessions, 2 will randomly allocate either 2 primary joint replacements or 1 revision joint replacement, while the third session will schedule 1 primary joint replacement."
 


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
	
	st.markdown('## Model execution')
	replications = st.slider('Multiple runs', 1, 50, 10)
	runtime = st.slider('Runtime (days)', 30, 100, 60)
  


schedule = md.Schedule()
args = md.Scenario(schedule)

args.n_beds = n_beds
args.primary_hip_mean_los = primary_hip_los
args.primary_knee_mean_los = primary_knee_los
args.revision_hip_mean_los = revision_hip_los
args.revision_knee_mean_los = revision_knee_los
args.unicompart_knee_mean_los = unicompart_knee_los
args.delay_post_los_mean = los_delay
args.prob_ward_delay = prop_delay
#scenario_schedule = st.session_state['SCENARIO_SCHEDULE_AVAIL']


if st.button('Start simulation'):
    # Get results
	with st.spinner('Simulating...'):
		m_results = md.multiple_reps(args, n_reps = replications, results_collection=runtime+md.DEFAULT_WARM_UP_PERIOD)[0]
		m_day_results = md.multiple_reps(args, n_reps = replications, results_collection=runtime+md.DEFAULT_WARM_UP_PERIOD)[1]
		m_primary_pt_results = md.multiple_reps(args, n_reps = replications, results_collection=runtime+md.DEFAULT_WARM_UP_PERIOD)[2]
		m_revision_pt_results = md.multiple_reps(args, n_reps = replications, results_collection=runtime+md.DEFAULT_WARM_UP_PERIOD)[3]
  
	# # save results to csv 
	# m_day_results.to_csv('data/day_results.csv')
	# m_primary_pt_results.to_csv('data/primary_patient_results.csv')
	# m_revision_pt_results.to_csv('data/revision_patient_results.csv')
	st.success('Done!')
	
	col1, col2 = st.columns(2)
	with col1:
		st.dataframe(m_results.head(3))
		st.dataframe(m_day_results.head(3))
		st.dataframe(m_primary_pt_results.head(3))
		st.dataframe(m_revision_pt_results.head(3))
	with col2:
		st.pyplot(md.weekly_summ_bed_utilisation(m_day_results))
		st.pyplot(md.daily_summ_bed_utilisation(m_day_results))
        
    
    







