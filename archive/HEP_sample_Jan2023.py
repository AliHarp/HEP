#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
from PIL import Image
import model as md


st.set_page_config(
     layout="wide",
     initial_sidebar_state="expanded"
 )
 
 
st.title(':green[Hospital Efficiency Project]')

st.header('Orthopaedic Planning Model')

image = Image.open('images/th-212327775.jpeg')
st.sidebar.image(image)

image = Image.open('images/University-of-Exeter.jpg')
st.sidebar.image(image)

image = Image.open('images/HdRlogo.jpg')
st.sidebar.image(image)

col1, col2 = st.columns(2)

with col1:

	"Hospital Efficiency Project is a 3-year, HDRUK funded project aimed at improving the efficiency of orthopaedic pathways. Is is a collaboration between \
	the Universities of Bristol and Exeter, and North Bristol NHS Trust."

	"This interactive tool is a discrete event simulation model, developed as a single component of the larger Hospital Efficiency Project multi-study."

	"The model represents total joint replacement activity from theatre scheduling, to ward stay, to discharge for primary and revision hip and knee replacement surgeries, \
	and unicompartmental knee replacement surgery."

with col2:
	image = Image.open('images/HEP_FLOW.jpg')
	st.image(image)
	
	image = Image.open('images/HEP_params.jpg')
	st.image(image)

with col1:

	"The theatre schedule is flexible. The number of sessions per weekday, and the surgical categories per session can be selected."

	"These can be compared with a default schedule in which 4 theatres operate 5 days per week, with 3 sessions per day.  Of those session, 2 will randomly \
	allocate either 2 primary joint replacements or 1 revision joint replacement, while the third session will schedule 1 primary joint replacement."

	"The flexible schedule will enable experimentation with the model, for example the effects of increasing the number of sessions one day per week, of adding \
	a weekend session, or of scheduling revision joint replacement surgery early in the week.  The impacts of these changes on bed utilisation and \
	total surgical throughput can be investigated."

	"Other scenarios can also be investigated in the model."

	"For example, the mean lengths-of-stay for surgical types, and the number of ringfenced beds available to patients can be changed to understand the how these\
	affect surgical throughput.  Additionally, the mean number of days delayed can be changed.  This represents a proportion of patients whose ward length-of-stay\
	is delayed for various reasons, such as awaiting a community package of care.  The proportion of these patients can also be changed."

	":green[The HEP project group consists of:]"
	":green[Rebecca Wilson, Maria Theresa Redaniel, Emily Eyles, Tim Jones, Chris Penfold, Ashley Blom, Andrew Elliott, Alison Harper, Tim Keen, Martin Pitt, Andrew Judge]"
	":green[With thanks to Mike Whitehead, Tom Woodward and Tom Monks for their support]"
	
	":green[This project is supported by the NIHR. The views expressed are those of the author(s) and not necessarily those of the NIHR or the Department of Health and Social Care.]"
	
	with col2:
		image = Image.open('images/nihr.png')
		st.image(image)
	







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

image2 = Image.open('images/funded-by-nihr-logo.png')
st.image(image2)

#hep.summary_frame = hep.scenario_summary_frame(hep.scenario_results)
#st.write(hep.summary_frame.round(2))

#hep.daily_audit = hep.scenario_daily_audit(hep.scenario_results[1])
#st.write(hep.daily_audit)

#hep.weekly_audit = hep.scenario_weekly_audit(hep.scenario_results[1])
#st.write(hep.weekly_audit)

#hep.lost_slots = hep.lost_slots(hep.patient_summary)
#st.write(hep.lost_slots)

n_beds = st.sidebar.slider('Beds', min_value = 20, max_value=60, value=40)
st.write('Set number of beds', n_beds)

n_theatres = st.sidebar.slider('Theatres', min_value = 2, max_value=6, value=4)
st.write('Set number of theatres', n_theatres)

n_opdays = st.sidebar.slider('Operating Days', min_value = 5, max_value=7, value=5)
st.write('Set number of theatre days', n_opdays)

primary_hip_los = st.sidebar.slider('Primary Hip LoS', min_value = 2, max_value = 5, value = 3)
st.write('Set primary hip LoS', primary_hip_los)

primary_knee_los = st.sidebar.slider('Primary Knee LoS', min_value = 2, max_value = 5, value = 4)
st.write('Set primary knee LoS', primary_knee_los)

revision_hip_los = st.sidebar.slider('Revision Hip LoS', min_value = 3, max_value = 6, value = 5)
st.write('Set revision hip LoS', revision_hip_los)

revision_knee_los = st.sidebar.slider('Revision Knee LoS', min_value = 3, max_value = 7, value = 5)
st.write('Set revision knee LoS', revision_knee_los)

unicompart_knee_los = st.sidebar.slider('Unicompart knee LoS', min_value = 2, max_value = 4, value = 3)
st.write('Set unicompart knee LoS', unicompart_knee_los)
    
st.markdown('## Scenarios Definitions')
