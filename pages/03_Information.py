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

st.header('Information')

image = Image.open('images/HdRlogo.jpg')
st.sidebar.image(image)
st.sidebar.markdown("***")

image = Image.open('images/th-212327775.jpeg')
st.sidebar.image(image)

image = Image.open('images/University-of-Exeter.jpg')
st.sidebar.image(image)

"Discrete event simulation is a computer model that mimics the operation of a real or proposed system or process."
"The simulation builds in the randomness seen in real life. For example, patients don't always stay for exactly 3 days after surgery."
"When the simulation runs, the interaction between resources and constraints can be seen as time passes."
	
"Making changes to the model gives an indication of how the system would behave in real life."
"This means ideas can be quickly tried out to gain insight into the effects of changes on the real system."

"Simulation models are an abstraction of a real system, and require some assumptions in translating the real system into a computer model."
"**The HEP model includes the following assumptions:**" 
st.markdown(" -  *Theatres only schedule the five main elective orthopaedic surgeries*")
st.markdown(" -  *The same scheduling rules and session numbers apply across all theatres*")
st.markdown(" -  *Beds are ring-fenced*")
st.markdown(" -  *Patient lengths-of-stay do not vary by day-of-week of admission*")
 

st.markdown("***")	


"The Hospital Efficiency Project simulation was developed using SimPy."
"Simpy is a process-based discrete-event simulation framework based on standard Python. SimPy is released under MIT License."
"SimPy documentation is available here:  https://simpy.readthedocs.io "
	
"This web-based application was developed using Streamlit, an open-source Python library for building and deploying data apps."
"Streamlit documentation is available here: https://docs.streamlit.io "

st.markdown("***")

"Hospital Efficiency Project is a 3-year, HDRUK funded project aimed at improving the efficiency of orthopaedic pathways."
"It is a collaboration between the Universities of Bristol and Exeter, and North Bristol NHS Trust."
	
":green[The HEP project group consists of:]"
":green[Alison Harper, Rebecca Wilson, Maria Theresa Redaniel, Emily Eyles, Tim Jones, Chris Penfold, Andrew Elliott, Tim Keen, Ashley Blom, Martin Pitt, Andrew Judge]"
":green[With thanks to Mike Whitehouse, Tom Woodward and Tom Monks for their support]"
	
":green[For further information, comments, or requests, please contact: Alison Harper a.l.harper@exeter.ac.uk]"
	
st.markdown("***")	
		
st.sidebar.markdown("***")	
st.sidebar.markdown(":blue[This project is supported by the NIHR. The views expressed are those of the author(s) and not necessarily those of the NIHR or the Department of Health and Social Care.]")
	

image = Image.open('images/nihr.png')
st.sidebar.image(image)
st.sidebar.markdown("***")		


