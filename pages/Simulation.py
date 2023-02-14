#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import model as md


st.set_page_config(
     layout="wide",
     initial_sidebar_state="expanded"
 )
 
 
st.title(':green[Hospital Efficiency Project]')

st.header('Orthopaedic Planning Model')

"The simulation model will use the schedule created on the previous page and compare it with a default schedule in which 4 theatres operate 5 days per week, with 3 sessions per day.  Of those session, 2 will randomly allocate either 2 primary joint replacements or 1 revision joint replacement, while the third session will schedule 1 primary joint replacement."

st.write(md.DEFAULT_NUMBER_BEDS)

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










st.markdown('## Scenarios Definitions')


data = pd.DataFrame({
  " ": [1,2,3,4,5,6],
  "Scenario": ['Current', 'Beds + 10', 'Theatres + 1', 'LoS + 20', 'Primaries + 1', 'Weekend + 1'],
  "Description": ['Uses default settings - represents how the system currently operates', 
  'Add an additional ten orthopaedic beds', 'Add an additional operating theatre', 
  'All surgery types have an extra 20 days LoS', 'Perform one extra primary joint replacement per theatre', 
  'Add Saturday operating']
})

st.write(data)

st.markdown('## Scenario summary results')

df = pd.read_csv("data/scenarios_summary_frame.csv")
st.write(df)

st.markdown('## Scenario results for bed utilisation per day of week')

image = Image.open('images/Mean daily bed utilisation scenarios.png')
st.image(image)

st.markdown('## Scenario results for lost slots per day of week')


#hep.summary_frame = hep.scenario_summary_frame(hep.scenario_results)
#st.write(hep.summary_frame.round(2))

#hep.daily_audit = hep.scenario_daily_audit(hep.scenario_results[1])
#st.write(hep.daily_audit)

#hep.weekly_audit = hep.scenario_weekly_audit(hep.scenario_results[1])
#st.write(hep.weekly_audit)

#hep.lost_slots = hep.lost_slots(hep.patient_summary)
#st.write(hep.lost_slots)

