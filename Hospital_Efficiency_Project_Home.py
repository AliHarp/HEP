#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
#from streamlit_state import SessionState

from PIL import Image


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
	st.write(':blue[Orthopaedic Model Process Flow]')
	image = Image.open('images/HEP_FLOW.jpg')
	st.image(image)
	
	st.write(':blue[Model Outputs and Inputs]')
	image = Image.open('images/HEP_params.jpg')
	st.image(image)

with col1:

	"The theatre schedule is flexible. The number of sessions per weekday, the surgical categories per session, and the number of operating theatres can be selected."

	"These can be compared with a default schedule in which 4 theatres operate 5 days per week, with 3 sessions per day.  Of those sessions, 2 will randomly \
	allocate either 2 primary joint replacements or 1 revision joint replacement, while the third session will schedule 1 primary joint replacement."

	"The flexible schedule will enable experimentation with the model, for example the effects of increasing the number of sessions per day of week, of adding \
	weekend sessions, or of scheduling different surgical types across weekdays."  
	
	"The impacts of these changes on bed utilisation and total surgical throughput can be investigated."

	"Other scenarios can also be investigated in the model."

	"For example, the mean lengths-of-stay of different surgical types, and the number of ringfenced beds available to patients can be changed to understand \
	effects on surgical throughput.  Additionally, the mean number of days delayed can be changed.  This represents a proportion of patients whose ward length-of-stay\
	is delayed for various reasons, such as awaiting a community package of care."
	  
	"The proportion of these patients can also be changed."

	":green[The HEP project group consists of:]"
	":green[Rebecca Wilson, Maria Theresa Redaniel, Emily Eyles, Tim Jones, Chris Penfold, Ashley Blom, Andrew Elliott, Alison Harper, Tim Keen, Martin Pitt, Andrew Judge]"
	":green[With thanks to Mike Whitehead, Tom Woodward and Tom Monks for their support]"
	
	":green[This project is supported by the NIHR. The views expressed are those of the author(s) and not necessarily those of the NIHR or the Department of Health and Social Care.]"
	
	with col2:
		image = Image.open('images/nihr.png')
		st.image(image)
	


