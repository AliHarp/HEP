#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import streamlit as st
import HEP_main as hep


st.title('Hospital Efficiency Project DES model')
st.title('Orthopaedic Planning Model')

# as well as rounding you may want to rename the cols/rows to 
# more readable alternatives.
hep.summary_frame = hep.scenario_summary_frame(hep.scenario_results)
st.write(hep.summary_frame.round(2))

n_beds = st.slider('Beds', min_value = 20, max_value=60, value=hep.args.n_beds)
st.write('Set number of beds', n_beds)
