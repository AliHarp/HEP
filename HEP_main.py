#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# ## Set start date and define it to be equal to the simulation time
# This allows monitoring of day-of-week.  
# 
# Defined in days: Monday = 0

# ## MIKE notes  
# list of actual times of surgery to see how many could fit in an actual schedule - (scenario) 
# 
# efficient use of time ie minimise downtime
# 
# ie v simple rule - question it
# Find out who the scheduler is and how she schedules

# In[2]:


start = arrow.get('2022-06-27')  #start on a monday -- ?more dynamic?
env = simpy.Environment()


# In[3]:


#checking - note start a/a not in real time.  Monday=0
current_date = start.shift(days=env.now)  
print('Current weekday:', current_date.weekday())

tomorrow_date = current_date.shift(days=+1)
print('Tomorrow weekday:', tomorrow_date.weekday())


start.shift(days=env.now).weekday()


# ## Output created: HEP_fit_delayed_LOS.docx
# > ### summary(primary_hip_v_los)  
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.   
#   1.000   3.000   4.000   5.747   7.000  29.000   
# > ### summary(primary_knee_v_los)  
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.   
#   1.000   3.000   4.000   5.386   6.000  28.000   
# > ### summary(revise_hip_v_los)  
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.   
#   1.000   4.000   7.000   9.149  13.000  29.000   
# > ### summary(revise_knee_v_los)  
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.   
#   1.000   4.000   7.000   9.022  13.000  29.000   
# > ### summary(uni_knee_v_los)  
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.   
#   1.000   2.000   3.000   3.389   4.000  28.000   
# > ### summary(delayed_los)  
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.   
#   30.00   34.00   43.00   49.61   58.00  156.00   
#   0        4       39      19.6    28     126  

# ## Parameters
# * Primary: primary hip, primary knee, primary uni-compartmental knee
# * Revision: revision hip, revision knee

# In[4]:


# ward parameters
primary_hip_mean_los = 4.747
primary_knee_mean_los =  4.386
revision_hip_mean_los = 6.149
revision_knee_mean_los = 6.022
unicompart_knee_mean_los = 2.389

primary_hip_sd_los = 2
primary_knee_sd_los = 2
revision_hip_sd_los = 3
revision_knee_sd_los = 3
unicompart_knee_sd_los = 1

delay_post_los_mean = 16.6
delay_post_los_sd = 10

prob_surgery_on_day = 0.95
prob_ward_delay = 0.05

#Ward resources
number_beds = 46

#patient parameters
#same day cancellations patient reasons
#prop_not_cancel_primary = 0.98
#prop_not_cancel_revision = 0.99

primary_dict = {1:'p_hip', 2:'p_knee', 3:'uni_knee'}
revision_dict = {1:'r_hip', 2:'r_knee'}
primary_prob = [0.4,0.4,0.2]
revision_prob = [0.6, 0.4]

#theatre resources
number_theatres = 4
primaries_per_day = [1,3,5] #primaries between 1 and 5, revisions dep. upon no. primaries
revisions_per_day = [2,1,0] #adjust for scenario: add extra theatre slot in

# skeleton frame for theatre schedule: default is no surgical activity on weekends
# for weekend activity, change '0' to '1'
schedule_list = {'Day':['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                        'Friday', 'Saturday', 'Sunday'],
                'Primary_slots':[1,1,1,1,1,0,0],
                'Revision_slots':[1,1,1,1,1,0,0]}

#simulation parameters
number_of_runs = 100
results_collection_period = 70  ## WILL EXCLUDE 0-DAY ARRIVALS AS A SIMULATION DAY
warm_up_period =  35 ## WILL EXCLUDE 0-DAY ARRIVALS AS A SIMULATION DAY
default_rng_set = None 
#results auditing
first_obs = 1
interval = 1

TRACE = False


# ## TOM NOTES
# 
# one long run 10 yrs p159 robinson, Banks et al  
# 
# sensitivity analysis - los params and distrib (lnorm, gamma, edf) +/- trunc  
# 
# outliers, double peak etc  
# 
# vary run lengths eg 1mth vs 1 year  
# 
# outputs by day and by patient  
# 
# 2k design, interaction - robinson, tom paper
# 
# MORE plots

# In[5]:


# later for translation: ignore this - testing

primary_dict = {1:'p_hip', 2:'p_knee', 3:'uni_knee'}
revision_dict = {1:'r_hip', 2:'r_knee'}
primary_prop = np.random.choice(np.arange(1,4), p=[0.4,0.4,0.2])
revision_prop = np.random.choice(np.arange(1,3), p=[0.6, 0.4])

def vec_tran(prop, dict):
    return np.vectorize(dict.__getitem__)(prop)

#trial sample and vectorize by dict key
primary_sample = vec_tran(primary_prop, primary_dict)
print(primary_sample)

print(int(primary_prop))
print(revision_prop)

type(primary_prop)


# ## Trace utility for debugging

# In[6]:


def trace(msg):
    """ 
    If TRUE will return all patient level message output
    """
    if TRACE:
        print(msg)


# ## Distribution classes for generating LoS and branching
# * Used to control random number sampling
# * Each distribution has its own stream
# * Lognormal for lengths-of-stay
# * Bernoulli for determining lost theatres slots and delayed discharges

# In[7]:


class Lognormal:
    """for creating LoS distributions for each patient type"""
    def __init__(self, mean, stdv, random_seed = None):
        self.rng = np.random.default_rng(seed = random_seed)
        mu, sigma = self.calc_params(mean, stdv)
        self.mu = mu
        self.sigma = sigma
        
    def calc_params(self, mean, stdv):
        phi = (stdv **2 + mean **2) **0.5
        mu = np.log(mean**2/phi)
        sigma = (np.log(phi **2/mean **2)) **0.5
        return mu, sigma
    
    def sample(self):
        """
        method to generate a sample from the lognormal distribution
        """
        return self.rng.lognormal(self.mu, self.sigma)
        
class Bernoulli:
    """for pathway branching: slots lost on day, 
    patients whose LoS is delayed due to downstream processes
    """
    def __init__(self, p, random_seed = None):
        """p = prob of drawing a 1"""
        self.rng = np.random.default_rng(seed=random_seed)
        self.p = p
      
    def sample(self, size = None):
        """
        method to generate a sample from the Bernoulli distribution
        """
        return self.rng.binomial(n = 1, p = self.p, size = size) 
    
class Gamma:
    """ sensitivity analysis on LoS distributions for each patient type"""
    def __init__(self, mean, stdv, random_seed = None):
        self.rng = np.random.default_rng(seed = random_seed)
        scale, shape = self.calc_params(mean, stdv)
        self.scale = scale
        self.shape = shape
        
    def calc_params(self, mean, stdv):
        scale = (stdv **2) / mean 
        shape = (stdv **2) / (scale **2)
        return scale, shape
    
    def sample(self, size = None):
        """
        method to generate a sample from the gamma distribution
        """
        return self.rng.gamma(self.shape, self.scale, size = size)   


# ## Theatre schedule
#  Default settings:
# * 4 theatres (2-6)
# * 5 day/week (5-7)
# * Each theatre has three sessions per day:
#         * Morning: 1 revision OR 2 primary
#         * Afternoon: 1 revision OR 2 primary
#         * Evening: 1 primary
#         

# In[8]:


def surgery_types(list): 
    """
    Randomly sample surgeries per day given 3* 4-hour sessions/day
    
    Number of revision surgeries depends upon number of primary surgeries
    
    * Each theatre has three sessions per day and some rules:
        * Morning: 1 revision OR 2 primary
        * Afternoon: 1 revision OR 2 primary
        * Evening: 1 primary
    """
    list = primaries_per_day
    select_primaries = random.choice(list)
    if select_primaries == primaries_per_day[0]:
        select_revisions = revisions_per_day[0]
    elif select_primaries == primaries_per_day[1]:
        select_revisions = revisions_per_day[1]
    else: select_revisions = revisions_per_day[2]
    
    return pd.Series([select_primaries, select_revisions])

def sample_week_schedule(surgery_types, schedule_list):
    """
    optional
    returns a plot of illustrative sample for one week, one theatre
    """
    schedule_list_sample = pd.DataFrame(schedule_list)
    schedule_list_sample[['Primary_slots', 'Revision_slots']] = \
        schedule_list_sample['Primary_slots'].apply(lambda x: x*surgery_types(list))
    schedule_list_plot = pd.DataFrame(schedule_list_sample).plot.bar(x ='Day'); 
    return schedule_list_plot

def theatre_capacity(surgery_types,schedule_list):
    """
    Calculate full theatre capacity schedule for given number of theatres
    Randomly create each theatre's schedule each day according to rules
    Result = 4-20 primary; 0-8 revision per day for default settings
    """
    schedule_avail = pd.DataFrame(schedule_list) 
    #skeleton schedule full length
    length_sched = int(round(2*(warm_up_period+results_collection_period)/7, 0))
    schedule_avail = schedule_avail.iloc[np.tile(np.arange(len(schedule_avail)), 
                        length_sched)].reset_index()
    #for calculating schedule
    schedule_avail_temp = schedule_avail.copy()
    schedule_avail_temp2 = schedule_avail.copy()

    schedule_avail['Primary_slots'].values[:] = 0
    schedule_avail['Revision_slots'].values[:] = 0
    for i in range(number_theatres):
        schedule_avail_temp[['Primary_slots', 'Revision_slots']] = \
            schedule_avail_temp2['Primary_slots'].\
            apply((lambda x: x*surgery_types(primaries_per_day)))
        schedule_avail[['Primary_slots', 'Revision_slots']] += \
            (schedule_avail_temp[['Primary_slots', 'Revision_slots']]) 
    return(schedule_avail)

#print sample plot if interested
sample_week_schedule(surgery_types, schedule_list)

#create full schedule, save to csv, print head to confirm
schedule_avail = theatre_capacity(surgery_types,schedule_list)
schedule_avail.to_csv('data/schedule2.csv')
schedule_avail.head(10)


# ## Scenarios class 

# In[9]:


class Scenario:
    """
    Holds LoS dists for each patient type
    Holds delay dists
    Holds prob of delay, prob of same day dist
    Holds resources: beds
    Passed to hospital model and process classes
    """
    def __init__(self, random_number_set=default_rng_set):
        """
        controls initial seeds of each RNS used in model
        """
        self.random_number_set = random_number_set
        self.init_sampling()
        self.init_resource_counts()
        
    def set_random_no_set(self, random_number_set):
        """
        controls random sampling for each distribution used in simulations"""
        self.random_number_set = random_number_set
        self.init_sampling()
        
    def init_resource_counts(self):
        """only one resource in the model: beds"""
        self.n_beds = number_beds
        
    def init_sampling(self):
        """
        distribs used in model and initialise seed"""
        rng_streams = np.random.default_rng(self.random_number_set)
        self.seeds = rng_streams.integers(0,99999999999, size = 20)
        
        #######  Distributions ########
        
        # LoS distribution for each surgery patient type
        self.primary_hip_dist = Lognormal(primary_hip_mean_los, primary_hip_sd_los,
                                          random_seed=self.seeds[0])
        self.primary_knee_dist = Lognormal(primary_knee_mean_los, primary_knee_sd_los,
                                          random_seed=self.seeds[1])
        self.revision_hip_dist = Lognormal(revision_hip_mean_los, revision_hip_sd_los,
                                          random_seed=self.seeds[2])
        self.revision_knee_dist = Lognormal(revision_knee_mean_los, revision_knee_sd_los,
                                          random_seed=self.seeds[3])
        self.unicompart_knee_dist = Lognormal(unicompart_knee_mean_los, unicompart_knee_sd_los,
                                          random_seed=self.seeds[4])
        
        # distribution for delayed LoS
        self.los_delay_dist = Lognormal(delay_post_los_mean, delay_post_los_sd,
                                       random_seed=self.seeds[5])
        
        # probability of no same day cancellations 
        self.prob_surgery_occurring = Bernoulli(prob_surgery_on_day, random_seed=self.seeds[6])
        
        #probability of having LoS delayed on ward
        self.los_delay = Bernoulli(prob_ward_delay, random_seed=self.seeds[7])
        
    def number_slots(self, schedule_avail):
        """
        convert to np arrays for each surgery type for patient generators
        """
        self.schedule_avail_primary = schedule_avail['Primary_slots'].to_numpy()
        self.schedule_avail_revision = schedule_avail['Revision_slots'].to_numpy()
        return(self.schedule_avail_primary, self.schedule_avail_revision)

    def primary_types(self,prob):
        """
        randomly select primary surgical type from custom distribution: primary_prop
        prob = primary_prop
        used for generating primary patients of each surgical type
        """
        self.primary_surgery = np.random.choice(np.arange(1,4), p=prob)
        return(self.primary_surgery)
    
    def revision_types(self,prob):
        """
        randomly select revision surgical type from custom distribution: revision_prop
        prob = revision_prop
        used for generating revision patients of each surgical type
        """
        self.revision_surgery = np.random.choice(np.arange(1,3), p=prob)
        return(self.revision_surgery)
     
    def label_types(self, prop, dict): 
        """
        return label for each surgery type
        """
        return np.vectorize(dict.__getitem__)(prop)


# In[10]:


prob = [0.4,0.6]
def revision_types_2(prob):
    revision_surgery2 = np.random.choice(np.arange(1,3), p=prob)
    return(revision_surgery2)

revision_types_2(prob)


# ## Set up process to get started: patient pathways
# Patient journeys for primary and revision patients
# 
# 
# 

# In[11]:


class PrimaryPatient:
    """
    The process a patient needing primary hip or knee surgery will undergo
    from scheduled admission for surgery to discharge
    
    day = simulation day
    id = patient id
    args: Scenario parameter class
    """
    def __init__(self, day, id, env, args):
        
        self.day = day
        self.id = id
        self.env = env
        self.args = args
        
        self.arrival = -np.inf
        self.queue_beds = -np.inf
        self.primary_los = 0
        self.total_time = -np.inf
        self.depart = -np.inf
        
        self.lost_slots_bool = False
        self.delayed_los_bool = False
        self.weekday = 0
        
    def service(self):
        """
        Arrive according to theatres schedule
        Some patients will leave on day of surgery and the slot is lost
        Some patients will have their surgery cancelled due to lack of beds
        Otherwise, patient is admitted and stays in a bed
        Some patients will have a post-bed request delay to their LoS
        Patient is discharged
        """
        
        self.arrival = self.env.now
        self.patient_class = 'primary'
        self.weekday = start.shift(days=self.env.now).weekday()
        
        # set los for primary surgery types
        self.types = int(self.args.primary_types(primary_prob))
        if self.types == 1:
            self.primary_los = self.args.primary_hip_dist.sample()
            self.primary_label = 'p_hip'
        elif self.types == 2:
            self.primary_los = self.args.primary_knee_dist.sample()
            self.primary_label = 'p_knee'
        else: 
            self.primary_los = self.args.unicompart_knee_dist.sample()
            self.primary_label = 'uni_knee'

        #vectorize according to dict key to get surgical type
        #self.primary_label = self.args.label_types(primary_prop, primary_dict)   
            
        #sample if need for delayed discharge
        self.need_for_los_delay = self.args.los_delay.sample()
        
        #Patients who have a delayed discharge follow this pathway
        if self.need_for_los_delay:
            
            #request a bed on ward - if none available within 0-0.25 day, patient has surgery cancelled
            with self.args.beds.request() as req:
                
                admission = random.uniform(0.25,0.5)
                admit = yield req | self.env.timeout(admission)

                if req in admit:
                    """record queue time for primary patients -- if > admission, 
                    this patient will leave the system and the slot is lost"""
                    
                    self.queue_beds = self.env.now - self.arrival
                    trace(f'primary patient {self.id} {self.primary_label}' 
                          f'has been allocated a bed at {self.env.now:.3f}' 
                          f'and queued for {self.queue_beds:.3f}')
                    
                    self.primary_los = self.primary_los + args.los_delay_dist.sample()
                    yield self.env.timeout(self.primary_los)
                    self.lost_slots_bool = False
                    self.delayed_los_bool = True
                    self.depart = self.env.now
                    trace(f'los of primary patient {self.id} completed at {self.env.now:.3f}')
                    self.total_time = self.env.now - self.arrival
                    trace(f'primary patient {self.id} {self.primary_label}'
                          f'total los = {self.total_time:.3f} with delayed discharge')
                else:
                    #patient had to leave as no beds were available on ward
                    self.no_bed_cancellation = self.env.now - self.arrival
                    trace(f'primary patient {self.id} {self.primary_label}'
                          f'had surgery cancelled after {self.no_bed_cancellation:.3f}')
                    self.queue_beds = self.env.now - self.arrival
                    self.total_time = self.env.now - self.arrival
                    self.primary_los = 0
                    self.lost_slots_bool = True
                    self.delayed_los_bool = False
                    self.depart = self.env.now
                    trace(f'primary patient {self.id} {self.primary_label}' 
                          f'recorded {self.lost_slots_bool}')
        #no delayed los
        else:
            #request a bed on ward - if none available within 0 - 0.25 day, patient has surgery cancelled
            with self.args.beds.request() as req:
                admission = random.uniform(0.25,0.5)
                admit = yield req | self.env.timeout(admission)
                self.no_bed_cancellation = self.env.now - self.arrival

                if req in admit:
                    #record queue time for primary patients -- if >1, this patient will leave the system and the slot is lost
                    self.queue_beds = self.env.now - self.arrival
                    trace(f'primary patient {self.id} {self.primary_label}'
                          f'has been allocated a bed at {self.env.now:.3f}'
                          f'and queued for {self.queue_beds:.3f}')

                    self.primary_los = self.primary_los
                    yield self.env.timeout(self.primary_los)
                    self.lost_slots_bool = False
                    self.delayed_los_bool = False
                    self.depart = self.env.now
                    trace(f'los of primary patient {self.id} {self.primary_label}'
                          f'completed at {self.env.now:.3f}')
                    self.total_time = self.env.now - self.arrival
                    trace(f'primary patient {self.id} {self.primary_label}'
                          f'total los = {self.total_time:.3f}')
                    
                else:
                    #patient had to leave as no beds were available on ward
                    trace(f'primary patient {self.id} {self.primary_label}'
                          f'had surgery cancelled after {self.no_bed_cancellation:.3f}')
                    self.queue_beds = self.env.now - self.arrival
                    self.total_time = self.env.now - self.arrival
                    self.primary_los = 0
                    self.lost_slots_bool = True
                    self.delayed_los_bool = False
                    self.depart = self.env.now
                    trace(f'primary patient {self.id} {self.primary_label}' 
                          f'recorded {self.lost_slots_bool}')
    
class RevisionPatient:
    """
    The process a patient needing revision hip or knee surgery will undergo
    from scheduled admission for surgery to discharge
    
    day = simulation day
    id = patient id
    args: Scenario parameter class
    """
    def __init__(self, day, id, env, args):
        
        self.day = day
        self.id = id
        self.env = env
        self.args = args
        
        self.arrival = -np.inf
        self.queue_beds = -np.inf
        self.revision_los = 0
        self.total_time = -np.inf
        self.depart = -np.inf
        
        self.lost_slots_bool = False
        self.delayed_los_bool = False
        self.weekday = 0
        
        
    def service(self):
        """
        Arrive according to theatres schedule
        Some patients will leave on day of surgery and the slot is lost
        Some patients will have their surgery cancelled due to lack of beds
        Otherwise, patient is admitted and stays in a bed
        Some patients will have a post-bed request delay to their LoS
        Patient is discharged
        """
     
        self.arrival = self.env.now
        self.patient_class = 'revision'
        self.weekday = start.shift(days=self.env.now).weekday()
        
        # set los for revision surgery types
        self.types = int(self.args.revision_types(revision_prob))
        if self.types == 1:
            self.revision_los = self.args.revision_hip_dist.sample()
            self.revision_label = 'r_hip'
        else: 
            self.revision_los = self.args.revision_knee_dist.sample()
            self.revision_label = 'r_knee'
            
        #vectorize according to dict key to get surgical type
        #self.revision_label = self.args.label_types(revision_prop, revision_dict) 
        
        #sample if need for delayed discharge
        self.need_for_los_delay = self.args.los_delay.sample()
        
        if self.need_for_los_delay:    
        
        #request bed on ward - if none available within 0-0.25 day, patient has surgery cancelled
            with self.args.beds.request() as req:
                admission = random.uniform(0,0.25)
                admit = yield req | self.env.timeout(admission)

                if req in admit:
                    #record queue time for primary patients -- if >admission, this patient will leave the system and the slot is lost
                    self.queue_beds = self.env.now - self.arrival
                    trace(f'revision patient {self.id} {self.revision_label}'
                          f'has been allocated a bed at {self.env.now:.3f}'
                          f'and queued for {self.queue_beds:.3f}')
                
                    self.revision_los = self.revision_los + args.los_delay_dist.sample()
                    yield self.env.timeout(self.revision_los)
                    self.lost_slots_bool = False
                    self.delayed_los_bool = True
                    self.depart = self.env.now
                    trace(f'los of revision patient {self.id} {self.revision_label}'
                          f'completed at {self.env.now:.3f}')
                    self.total_time = self.env.now - self.arrival
                    trace(f'revision patient {self.id} {self.revision_label}'
                          f'total los = {self.total_time:.3f} with delayed discharge')

                else:
                    #patient had to leave as no beds were available on ward
                    self.no_bed_cancellation = self.env.now - self.arrival
                    trace(f'revision patient {self.id}'
                          f'had surgery cancelled after {self.no_bed_cancellation:.3f}')
                    self.queue_beds = self.env.now - self.arrival
                    self.total_time = self.env.now - self.arrival
                    self.revision_los = 0
                    self.lost_slots_bool = True
                    self.delayed_los_bool = False
                    self.depart = self.env.now
                    trace(f'revision patient {self.id} {self.revision_label}'
                          f'recorded {self.lost_slots_bool}')

        #no need for delayed discharge            
        else:
            #request bed on ward - if none available within 0-1 day, patient has surgery cancelled
            with self.args.beds.request() as req:
                admission = random.uniform(0,0.25)
                admit = yield req | self.env.timeout(admission)
                self.no_bed_cancellation = self.env.now - self.arrival

                if req in admit:
                    #record queue time for primary patients -- if >1, this patient will leave the system and the slot is lost
                    self.queue_beds = self.env.now - self.arrival
                    trace(f'revision patient {self.id} {self.revision_label}'
                          f'has been allocated a bed at {self.env.now:.3f}'
                          f'and queued for {self.queue_beds:.3f}')
                    self.revision_los = self.revision_los
                    yield self.env.timeout(self.revision_los)
                    self.lost_slots_bool = False
                    self.delayed_los_bool = False
                    self.depart = self.env.now

                    trace(f'los of revision patient {self.id} completed at {self.env.now:.3f}')
                    self.total_time = self.env.now - self.arrival
                    trace(f'revision patient {self.id} total los = {self.total_time:.3f}')

                else:
                    #patient had to leave as no beds were available on ward
                    trace(f'revision patient {self.id} {self.revision_label}'
                          f'had surgery cancelled after {self.no_bed_cancellation:.3f}')
                    self.queue_beds = self.env.now - self.arrival
                    self.total_time = self.env.now - self.arrival
                    self.revision_los = 0
                    self.lost_slots_bool = True
                    self.delayed_los_bool = False
                    self.depart = self.env.now 
                    trace(f'revision patient {self.id} {self.revision_label}' 
                          f'recorded {self.lost_slots_bool}')

    


# ## Monitor lost slots, beds occupied and throughput by day
# 

# In[12]:


class Hospital:
    """
    The orthopaedic hospital model
    """
    def __init__(self, args):
        self.env = simpy.Environment()
        self.args = args
        self.init_resources()
        
        #patient generator lists
        self.patients = []
        self.primary_patients = []
        self.revision_patients = []
        self.primary_patients_id = []
        self.revision_patients_id = []
        self.cum_primary_patients = []
        self.cum_revision_patients = []
               
        self.results_collection_period = None
        self.summary_results = None
        self.audit_interval = interval
        
        #lists used for daily audit_frame for summary results per day
        self.audit_time = []
        self.audit_day_of_week = []  
        self.audit_beds_used = []
        self.audit_primary_arrival = []
        self.audit_revision_arrival = []
        self.audit_primary_queue_beds = []
        self.audit_revision_queue_beds = []
        self.audit_primary_los = []
        self.audit_revision_los = []

        self.results = pd.DataFrame()
       
    def audit_frame(self):
        """
        Dataframe with results summarised per day 
        """
        self.results = pd.DataFrame({'sim_time':self.audit_time,
                                     'weekday': self.audit_day_of_week,
                                     'bed_utilisation': self.audit_beds_used,
                                     'primary_arrivals': self.audit_primary_arrival,
                                     'revision_arrivals': self.audit_revision_arrival,
                                     'primary_bed_queue': self.audit_primary_queue_beds,
                                     'revision_bed_queue': self.audit_revision_queue_beds,
                                     'primary_mean_los': self.audit_primary_los,
                                     'revision_mean_los': self.audit_revision_los})

    def patient_results(self):
        """
        Dataframes to hold individual results per patient per day per run
        Attributes from patient classes
        """
        
        results_primary_pt = pd.DataFrame({'Day':np.array([getattr(p, 'day') for p in self.cum_primary_patients]),
                             'weekday':np.array([getattr(p, 'weekday') for p in self.cum_primary_patients]),
                             'ID':np.array([getattr(p, 'id') for p in self.cum_primary_patients]),
                             'arrival time':np.array([getattr(p, 'arrival') for p in self.cum_primary_patients]),
                             'patient class':np.array([getattr(p, 'patient_class') for p in self.cum_primary_patients]),
                             'surgery type':np.array([getattr(p, 'primary_label') for p in self.cum_primary_patients]),
                             'lost slots':np.array([getattr(p, 'lost_slots_bool') for p in self.cum_primary_patients]),
                             'queue time':np.array([getattr(p, 'queue_beds') for p in self.cum_primary_patients]),
                             'los':np.array([getattr(p, 'primary_los') for p in self.cum_primary_patients]),
                             'delayed discharge':np.array([getattr(p, 'delayed_los_bool') for p in self.cum_primary_patients]),
                             'depart':np.array([getattr(p, 'depart') for p in self.cum_primary_patients])
                            })
    
        results_revision_pt = pd.DataFrame({'Day':np.array([getattr(p, 'day') for p in self.cum_revision_patients]),
                             'ID':np.array([getattr(p, 'id') for p in self.cum_revision_patients]),
                             'weekday':np.array([getattr(p, 'weekday') for p in self.cum_revision_patients]),
                             'arrival time':np.array([getattr(p, 'arrival') for p in self.cum_revision_patients]),
                             'patient class':np.array([getattr(p, 'patient_class') for p in self.cum_revision_patients]),
                             'surgery type':np.array([getattr(p, 'revision_label') for p in self.cum_revision_patients]),
                             'lost slots':np.array([getattr(p, 'lost_slots_bool') for p in self.cum_revision_patients]),
                             'queue time':np.array([getattr(p, 'queue_beds') for p in self.cum_revision_patients]),
                             'los':np.array([getattr(p, 'revision_los') for p in self.cum_revision_patients]),
                             'delayed discharge':np.array([getattr(p, 'delayed_los_bool') for p in self.cum_revision_patients]),
                             'depart':np.array([getattr(p, 'depart') for p in self.cum_revision_patients])
                            })
        return(results_primary_pt, results_revision_pt)
        
    def plots(self):
        """
        plot results at end of run
        """
    def perform_audit(self):
        """
        Results per day
        monitor ED each day and return daily results for metrics in audit_frame
        """
        yield self.env.timeout(warm_up_period)
        
        while True:
            #simulation time
            t = self.env.now
            self.audit_time.append(t)
            
            #weekday
            self.audit_day_of_week.append(start.shift(days=self.env.now -1).weekday())
            
            ##########  bed utilisation
            (self.audit_beds_used.append(self.args.beds.count / self.args.n_beds))
            
            ###########  lost slots
            patients = self.cum_revision_patients + self.cum_primary_patients
            
            # deal with lost slots on zero arrival days
            """
            lost_slots = []
            def zero_days(ls):
                if not zero_days:
                    return 1
                else:
                    return 0
 
            ls = (np.array([getattr(p, 'lost_slots_int') for p in patients]))
            if zero_days(ls):
                lost_slots = 0
            else:
            
            lost_slots = len(np.array([getattr(p,'lost_slots_int') for p in patients / len(patients)
            self.audit_slots_lost.append(lost_slots)
            """
            ######### arrivals
            pp = len(np.array([p.id for p in self.cum_primary_patients]))
            rp = len(np.array([p.id for p in self.cum_revision_patients]))
            self.audit_primary_arrival.append(len(self.primary_patients))
            self.audit_revision_arrival.append(len(self.revision_patients))
                                               
            #queue times
            primary_q = np.array([getattr(p, 'queue_beds') for p in self.cum_primary_patients
                                           if getattr(p, 'queue_beds') > -np.inf]).mean()
            self.audit_primary_queue_beds.append(primary_q)
                                               
            revision_q = np.array([getattr(p, 'queue_beds') for p in self.cum_revision_patients
                                           if getattr(p, 'queue_beds') > -np.inf]).mean()
            self.audit_revision_queue_beds.append(revision_q)
                                               
            #mean lengths of stay
            primarylos = np.array([getattr(p, 'primary_los') for p in self.cum_primary_patients
                                           if getattr(p, 'primary_los') > -np.inf]).mean().round(2)
            self.audit_primary_los.append(primarylos)
                                               
            revisionlos = np.array([getattr(p, 'revision_los') for p in self.cum_revision_patients
                                           if getattr(p, 'revision_los') > -np.inf]).mean().round(2)
            self.audit_revision_los.append(revisionlos)
            
            yield self.env.timeout(self.audit_interval)

    def init_resources(self):
        """
        ward beds initialised and stored in args
        """
        self.args.beds = simpy.Resource(self.env, 
                                        capacity=self.args.n_beds)
        
    def run(self, results_collection = results_collection_period+warm_up_period):
        """
        single run of model
        """
        self.env.process(self.patient_arrivals_generator_primary())
        self.env.process(self.patient_arrivals_generator_revision())
        self.env.process(self.perform_audit())
        self.results_collection = results_collection
        self.env.run(until=results_collection)
        audit_frame = self.audit_frame()
        return audit_frame
    
    def patient_arrivals_generator_primary(self):
        """
        Primary patients arrive according to daily theatre schedule
        ------------------
        """
        sched = args.number_slots(schedule_avail)[0]
        pt_count = 1
        for day in range(len(sched)):
            
            primary_arrivals = sched[day]
            trace(f'--------- {primary_arrivals} primary patients are scheduled on Day {day} -------')
            for i in range(primary_arrivals):
                
                    new_primary_patient = PrimaryPatient(day, pt_count, self.env, self.args)
                    self.cum_primary_patients.append(new_primary_patient)
                    self.primary_patients.append(new_primary_patient)
                    #for debuggng
                    self.primary_patients_id.append(new_primary_patient.id)
                    trace(f'primary patient {pt_count} arrived on day {day:.3f}')
                    self.env.process(new_primary_patient.service())
                    pt_count += 1
                    trace(f'primary ids: {self.primary_patients_id}')
            yield self.env.timeout(1)
            self.primary_patients *= 0
                    
            
    def patient_arrivals_generator_revision(self):
        """
        Revision patients arrive according to daily theatre schedule
        ------------------
        """    
        sched = args.number_slots(schedule_avail)[1]
        pt_count = 1
        for day in range(len(sched)):
            
            revision_arrivals = sched[day]
            trace(f'--------- {revision_arrivals} revision patients are scheduled on Day {day} -------')
            for i in range(revision_arrivals):
                    new_revision_patient = RevisionPatient(day, pt_count, self.env, self.args)
                    self.cum_revision_patients.append(new_revision_patient)
                    self.revision_patients.append(new_revision_patient)
                    #for debugging
                    self.revision_patients_id.append(new_revision_patient.id)
                    trace(f'revision patient {pt_count} arrived on day {day:.3f}')
                    self.env.process(new_revision_patient.service())
                    pt_count += 1
                    trace(f'revision ids: {self.revision_patients_id}')
            yield self.env.timeout(1)
            self.revision_patients *= 0
                    
                    


# ## Summary results across days and runs

# In[13]:


class Summary:
    """
    summary results across run
    """
    def __init__(self, model):
        """ model: Hospital """
        
        self.model = model
        self.args = model.args
        self.summary_results = None
        
    def process_run_results(self):
        self.summary_results = {}
        
        #all patients arrived
        patients = self.model.cum_primary_patients + self.model.cum_revision_patients

        #throughput
        primary_throughput = len([p for p in self.model.cum_primary_patients if (p.total_time > -np.inf)
                                  & (p.day > warm_up_period)])
        revision_throughput = len([p for p in self.model.cum_revision_patients if (p.total_time > -np.inf)
                                   & (p.day > warm_up_period)])

        #mean queues - this also includes patients who renege and therefore have 0 queue
        mean_primary_queue_beds = np.array([getattr(p, 'queue_beds') for p in self.model.cum_primary_patients
                                            if getattr(p, 'queue_beds') > -np.inf]).mean()
        mean_revision_queue_beds = np.array([getattr(p, 'queue_beds') for p in self.model.cum_revision_patients
                                            if getattr(p, 'queue_beds') > -np.inf]).mean()

        #check mean los
        mean_primary_los = np.array([getattr(p, 'primary_los') for p in self.model.cum_primary_patients
                                               if getattr(p, 'primary_los') > -np.inf]).mean()
        mean_revision_los = np.array([getattr(p, 'revision_los') for p in self.model.cum_revision_patients
                                               if getattr(p, 'revision_los') > -np.inf]).mean()

            ##############check these#############
        los_primary = np.array([getattr(p,'primary_los') for p in self.model.cum_primary_patients
                                if (getattr(p, 'primary_los') > -np.inf) & (getattr(p, 'day') > warm_up_period)]).sum()
        mean_primary_bed_utilisation = los_primary / (results_collection_period * self.args.n_beds)
        los_revision = np.array([getattr(p,'revision_los') for p in self.model.cum_revision_patients
                                if (getattr(p, 'revision_los') > -np.inf) & (getattr(p, 'day') > warm_up_period)]).sum()
        mean_revision_bed_utilisation = los_revision / (results_collection_period * self.args.n_beds)

        #bed_utilisation = (self.args.beds.count / self.args.n_beds)

        self.summary_results = {'arrivals':len(patients),
                                'primary_arrivals':len(self.model.primary_patients_id),  
                                'revision_arrivals':len(self.model.revision_patients_id),                     
                                'primary_throughput':primary_throughput,
                                'revision_throughput':revision_throughput,
                                'primary_queue':mean_primary_queue_beds,
                                'revision_queue':mean_revision_queue_beds,
                                'mean_primary_los':mean_primary_los,
                                'mean_revision_los':mean_revision_los,
                                'primary_bed_utilisation':mean_primary_bed_utilisation,
                                'revision_bed_utilisation':mean_revision_bed_utilisation}
    
    def summary_frame(self):
        if self.summary_results is None:
            self.process_run_results()
        df = pd.DataFrame({'1':self.summary_results})
        df = df.T
        df.index.name = 'rep'
        return df
                                            


# ## Functions for running the model and collecting the results

# In[14]:


def single_run(scenario, results_collection=results_collection_period+warm_up_period, random_no_set=default_rng_set):
    """
    summary results for a single run which can be called for multiple runs
    1. summary of single run
    2. daily audit of mean results per day
    3a. primary patient results for one run and all days
    3b. revision patient results for one run and all days
    """
    scenario.set_random_no_set(random_no_set)
    model = Hospital(scenario)
    model.run(results_collection = results_collection)
    summary = Summary(model)
    
    #summary results for a single run 
    #(warmup excluded apart from bed utilisation AND throughput)
    summary_df = summary.summary_frame()
    
    #summary per day results for a single run (warmup excluded)
    results_per_day = model.results
    
    #patient-level results (includes warmup results)
    patient_results = model.patient_results()
    
    return(summary_df, results_per_day, patient_results)

def multiple_reps(scenario, results_collection=results_collection_period+warm_up_period, n_reps=number_of_runs):
    """
    create dataframes of summary results across multiple runs:
    1. summary table per run
    2. summary table per run and per day
    3a. primary patient results for all days and all runs 
    3b. revision patient results for all days and all runs
    """
    #summary per run for multiple reps 
    #(warm-up excluded apart from bed utilisation AND throughput)
    results = [single_run(scenario, results_collection, random_no_set=rep)[0]
                         for rep in range(n_reps)]
    df_results = pd.concat(results)
    df_results.index = np.arange(1, len(df_results)+1)
    df_results.index.name = 'rep'
    
    #summary per day per run for multiple reps (warmup excluded)
    day_results = [single_run(scenario, results_collection, random_no_set=rep)[1]
                         for rep in range(n_reps)]
    
    length_run = [*range(1, results_collection-warm_up_period+1)]
    length_reps = [*range(1, n_reps+1)]
    run = [rep for rep in length_reps for i in length_run]
    
    df_day_results = pd.concat(day_results)
    df_day_results['run'] = run
    
    #patient results for all days and all runs (warmup included)
    primary_pt_results = [single_run(scenario, results_collection, random_no_set=rep)[2][0]
                         for rep in range(n_reps)]       
    primary_pt_results = pd.concat(primary_pt_results)

    revision_pt_results = [single_run(scenario, results_collection, random_no_set=rep)[2][1]
                         for rep in range(n_reps)]
    revision_pt_results = pd.concat(revision_pt_results)

    return (df_results, df_day_results, primary_pt_results, revision_pt_results)


# ## A single run

# In[15]:


# a single run
args = Scenario()
s_results = single_run(args, random_no_set = 42)
print(repr(s_results[0].T))
print(repr(s_results[1].head()))
print(repr(s_results[2][0].head()))
print(repr(s_results[2][1].head()))


# ## Multiple runs

# In[16]:


get_ipython().run_cell_magic('time', '', "args = Scenario()\nm_results = multiple_reps(args, n_reps=number_of_runs)[0]\nm_day_results = multiple_reps(args, n_reps=number_of_runs)[1]\nm_primary_pt_results = multiple_reps(args, n_reps=number_of_runs)[2]\nm_revision_pt_results = multiple_reps(args, n_reps=number_of_runs)[3]\n  \n# save results to csv \nm_day_results.to_csv('data/day_results.csv')\nm_primary_pt_results.to_csv('data/primary_patient_results.csv')\nm_revision_pt_results.to_csv('data/revision_patient_results.csv')\n\n# check outputs\nprint(repr(m_results.head(3)))\nprint(repr(m_day_results.head(3)))\nprint(repr(m_primary_pt_results.head(3)))\nprint(repr(m_revision_pt_results.head(3)))\n")


# ## Summary results overall for multiple runs

# In[17]:


def summary_over_runs(m_results):
    """
    summary results for multiple runs
    throughput and bed utilisation excludes results warm-up - arrivals include warmup
    visualise replications for throughput and utilisation
    """
    summ = m_results.mean().round(2)
    fig, ax = plt.subplots(4,1, figsize=(12,10))
    ax[0].hist(m_results['primary_bed_utilisation']);
    ax[0].set_ylabel('Primary bed utilisation')
    ax[1].hist(m_results['revision_bed_utilisation']);
    ax[1].set_ylabel('Revision bed utilisation')
    ax[2].hist(m_results['primary_throughput']);
    ax[2].set_ylabel('Primary throughput')
    ax[3].hist(m_results['revision_throughput']);
    ax[3].set_ylabel('Revision throughput')
    return(summ, fig)

summary_over_runs(m_results)


# # Summary results per day for multiple runs for bed utilisation
# 
# ## 1. Group by simulation time (day) across all runs

# In[20]:


def daily_summ_bed_utilisation(m_day_results): 
    """
    summarise per day across runs and save to csv in case of further analysis
    print bed utilisation plot
    warm-up results are excluded at runtime
    """
    m_day_results_ts = m_day_results.groupby(['sim_time']).mean()
    m_day_results_ts.to_csv('data_summaries/audit_day_results_across_runs.csv')
    fig, ax = plt.subplots(figsize=(22,3))
    ax.plot(m_day_results_ts['bed_utilisation'])
    ax.set_title('Bed Utilisation across model runtime (days)')
    ax.set_ylabel('Mean daily proportion of bed utilisation')
    return(fig)

daily_summ_bed_utilisation(m_day_results);


# # Summary results per day for multiple runs  
# 
# ## 2. Group by weekday

# In[21]:


def weekly_summ_bed_utilisation(m_day_results): 
    """
    summarise per week across runs and save to csv in case of further analysis
    print bed utilisation plot
    warm-up results are excluded at runtime
    """
    m_day_results_wd = m_day_results.groupby(['weekday']).mean()
    m_day_results_wd.to_csv('data_summaries/audit_weekday_results_across_runs.csv')
    values = m_day_results_wd['bed_utilisation']
    names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(names, values)
    ax.set_title('Mean bed Utilisation per day of week')
    return(fig)

weekly_summ_bed_utilisation(m_day_results);


# # Patient level results summarised by day and weekday
# 
# ## Lost slots calculation and plots per day and weekday
# 

# In[24]:


def calc_lost_theatre_slots(primary_pt_results, revision_pt_results):
    """
    Join pt dataframes into single df
    Select columns for lost slots
    Summarise by day across runs
    Deal with 0-day arrivals
    Save to csv
    Return lost_slots_df
    """
    pt_results = pd.concat([primary_pt_results, revision_pt_results])
    lost_slots_df = pt_results[["Day", "lost slots", "weekday"]]
    lost_slots_df = pd.DataFrame(lost_slots_df.groupby(['Day', 'weekday'])['lost slots'].sum().astype(int))
    lost_slots_df = lost_slots_df.assign(DayLostSlots = lambda x: (x['lost slots'] / number_of_runs))
    lost_slots_df = pd.DataFrame(lost_slots_df["DayLostSlots"]).reset_index()
    #0-arrival days excluded from df - add to Days sequence and fill lost slots value with 0 lost slots
    # re-index as dataframe length increasing. Fill values in columns with 0.
    lost_slots_df = (lost_slots_df.set_index('Day')
     .reindex(range(lost_slots_df.Day.iat[0],lost_slots_df.Day.iat[-1]+1), fill_value=0)
     .reset_index())
    #change 0 weekdays into correct weekday integer
    #need days of week seq and length of total range > length of dataframe
    shortseq = np.arange(len(range(0,7)))
    length = math.ceil(len(lost_slots_df) / 7)
    #create total sequence and flatten array list into list of elements
    sequence = ([np.tile((shortseq),length)])
    flat_seq = list(itertools.chain(*sequence))
    #truncate to correct length and save to column
    sequence = flat_seq[:len(lost_slots_df)]
    lost_slots_df['weekday'] = sequence
    lost_slots_df.to_csv('data_summaries/Lost_slots_results_per_day.csv')
    return(pd.DataFrame(lost_slots_df))


lost_slots_df = calc_lost_theatre_slots(m_primary_pt_results, m_revision_pt_results)


# In[25]:


def plot_lost_slots_per_day(lost_slots_df):
    """
    Remove warm-up period results
    Plot lost slots per day
    """
    lost_slots_df = lost_slots_df[lost_slots_df["Day"] > warm_up_period]
    fig, ax = plt.subplots(figsize=(22,3))
    ax.plot(lost_slots_df['DayLostSlots'])
    ax.set_title('Lost theatre slots across model runtime (days)')
    return(fig)
    

plot_lost_slots_per_day(lost_slots_df);


# In[26]:


def plot_lost_slots_per_week(lost_slots_df):
    """
    Remove warm-up period results
    Group by week
    """
    lost_slots_df = lost_slots_df[lost_slots_df["Day"] > warm_up_period]    
    lost_slots_wk_plot = lost_slots_df.groupby('weekday').mean()
    lost_slots_wk_plot.reset_index()
    values = lost_slots_wk_plot['DayLostSlots']
    names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(names, values)
    ax.set_title('Mean lost slots per day of week')
    return(fig)

plot_lost_slots_per_week(lost_slots_df);


# ## MORE plots 
# 
# * Determine appropriate replications

# In[27]:


#calc means of multiple reps by rep
more_plot_results = m_day_results.groupby(['run']).mean()
more_plot_results_ = more_plot_results.loc[:, ['bed_utilisation','primary_bed_queue', 'revision_bed_queue', 
                                               'primary_mean_los', 'revision_mean_los']] 
more_plot_results_ = more_plot_results_.reset_index(drop=True)


more_plot_results_ *= 100
more_plot_results_.columns = [0,1,2,3,4]
more_plot_results_.index.name = None

more_plot_results_.head()


# In[28]:


def ci_for_sample_mean(mean_value, std, n, critical_value=1.96):
    '''Confidence interval for mean.  Assume std is sample std.
    
    Notes:
    ------
    
    critical value hard coded at the moment.  
    Should update to use t dist.
    '''
    half_width = (critical_value * (std / np.sqrt(n)))
    mean_lower = mean_value - half_width
    mean_upper = mean_value + half_width
    return mean_lower, mean_upper

def ci_percentile(results, field, percentile, critical_value=1.96):
    '''Approximate confidence interval for percentile.
    Note these may or may not be symmetric.
    
    Notes:
    ------
    
    critical value hard coded at the moment.  
    Should update to use t dist.
    
    Params:
    ------
    results: pd.DataFrame
        Results dataframe - tabular data where each row is a rep and each col is a KPI
        
    field: int
        Field from data frame to analyse
        
    percentile: float
        The percentile around which to form the CI
        
    critical_value: float, optional (default = 1.96)
        critical value of the normal dist to use.
    '''
    half_width = critical_value * np.sqrt((percentile * (1 - percentile)) / (len(results) - 1))
    y_beta_1 = results[field].quantile(percentile - half_width)
    y_beta_2 = results[field].quantile(percentile + half_width)
    return y_beta_1, y_beta_2

def as_horizontal_axis_fraction(value, xmin, xmax):
    '''Convert a axis value to a fraction accounting for the 
    minimum on the xaxis (i.e. axis may not start from 0).
    '''
    return (value - xmin) / (xmax - xmin)

def draw_interval(ax, start, end, style="|-|", lw=3, color='b'):
    '''Annotate a matplotlib chart underneath x axis with an confidence interval.
    '''
    _ = ax.annotate('', xy=(start, -0.1), xycoords='axes fraction', 
                       xytext=(end, -0.1),
                       arrowprops=dict(arrowstyle=style, color=color, lw=lw))


# In[29]:


def more_plot(results, field=0, bins=None, figsize=(8, 5), percentiles=(0.05, 0.95), surpress_warnings=False):
    '''Measure of Risk and Error (MORE) plot.
    
    Risk illustrated via likely and unlikely ranges of replication values. 
    Erorr illustrated for CIs for mean and wide approx confidence intervals for percentiles
        
    Confidence intervals for percentiles will only be calculated if > 80 replications due to 
    approximation accuracy.
    
    Notes:
    ------
    Each value plotted represents the mean of a replication (e.g. daily throughput).  It should
    not be confused with an individuals results (e.g. an individuals throughput time). 
    
    If the system modelled contains time dependency the MORE plot may hide time of day/event effects.
    
    Params:
    ------
    results: pd.DataFrame
        Tabular data of replications. each column is a kpi
        
    field: int
        ID of column containing relevant data
        
    bins: int, optional (default=None)
        no. bins to generate. None=pandas decides no.
        
    figsize: tuple, optional (default=(8,5))
        size of plot
        
    
    Returns:
    -------
    fig, ax
    
    Refs:
    -----
    
    Nelson 2008. (Winter Simulation Paper)
    https://ieeexplore.ieee.org/document/4736095    
    
    '''
    
    # probably will shift these to module level scope.
    LIKELY = 'LIKELY'
    UNLIKELY = 'UNLIKELY'
    FONT_SIZE = 12
    LINE_WIDTH = 3
    LINE_STYLE = '-'
    CRIT_VALUE = 1.96
    UPPER_QUANTILE = percentiles[1]
    LOWER_QUANTILE = percentiles[0]
    INTERVAL_LW = 2
    MIN_N_FOR_PERCENTILE = 80
    WARN = f'CIs for percentiles are not generated as sample size < {MIN_N_FOR_PERCENTILE}.'
    WARN += ' To supress this msg set `supress_warnings=True`'

    ax = results[field].hist(bins=bins, figsize=figsize)
    mean = results[field].mean()
    std = results[field].std(ddof=1)
    upper_percentile = results[field].quantile(UPPER_QUANTILE)
    lower_percentile = results[field].quantile(LOWER_QUANTILE)

    # vertical lines
    ax.axvline(x=mean, linestyle='-', color='black', linewidth=LINE_WIDTH)
    ax.axvline(x=upper_percentile, linestyle='-', color='red', linewidth=LINE_WIDTH)
    ax.axvline(x=lower_percentile, linestyle='-', color='red', linewidth=LINE_WIDTH)

    like_font = {'family': 'serif',
                 'color':  'black',
                 'weight': 'bold',
                 'size': FONT_SIZE
                 }
    unlike_font = {'family': 'serif',
                 'color':  'red',
                 'weight': 'bold',
                 'size': FONT_SIZE
                 }

    # add text
    txt_offset = ax.get_ylim()[1] * 1.05
    ax.text(mean - (mean * 0.001), txt_offset, LIKELY, fontdict=like_font)
    ax.text(upper_percentile, txt_offset, UNLIKELY, fontdict=unlike_font)
    ax.text(ax.get_xlim()[0], txt_offset, UNLIKELY, fontdict=unlike_font)

    # calculate and display confidence intervals

    ## CIs for sample mean
    mean_lower, mean_upper = ci_for_sample_mean(mean, std, len(results))
    
    # Draw Confidence intervals
    # The horizontal lines are expressed as an axis fraction i.e. between 0 and 1.  
    # This means thatthe percentile CIs need to be converted before plotting.
    # The function as_horizontal_axis_fraction is used.

    ## mean CI  
    hline_mean_from = as_horizontal_axis_fraction(mean_lower, ax.get_xlim()[0], ax.get_xlim()[1])
    hline_mean_to = as_horizontal_axis_fraction(mean_upper, ax.get_xlim()[0], ax.get_xlim()[1])
    draw_interval(ax, hline_mean_from, hline_mean_to, lw=INTERVAL_LW)
    
    # avoid approximation issues with small samples.  
    if len(results) >= MIN_N_FOR_PERCENTILE:
        ## upper percentile
        y_beta_1, y_beta_2 = ci_percentile(results, field, UPPER_QUANTILE, critical_value=CRIT_VALUE)

        ## lower percentile
        y_beta_l_1, y_beta_l_2 = ci_percentile(results, field, LOWER_QUANTILE, critical_value=CRIT_VALUE)
        
        ## line for upper quantile CI
        hline_upper_q_from = (y_beta_1 - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
        hline_upper_q_to = (y_beta_2 - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
        
        hline_upper_q_from = as_horizontal_axis_fraction(y_beta_1, ax.get_xlim()[0], ax.get_xlim()[1])
        hline_upper_q_to = as_horizontal_axis_fraction(y_beta_2, ax.get_xlim()[0], ax.get_xlim()[1])
        draw_interval(ax, hline_upper_q_from, hline_upper_q_to, lw=INTERVAL_LW)
        
        ## line for lower quantile CI
        hline_lower_q_from = (y_beta_l_1 - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
        hline_lower_q_to = (y_beta_l_2 - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
        
        hline_lower_q_from = as_horizontal_axis_fraction(y_beta_l_1, ax.get_xlim()[0], ax.get_xlim()[1])
        hline_lower_q_to = as_horizontal_axis_fraction(y_beta_l_2, ax.get_xlim()[0], ax.get_xlim()[1])
        draw_interval(ax, hline_lower_q_from, hline_lower_q_to, lw=INTERVAL_LW)
        
    elif not surpress_warnings:
        warnings.warn(WARN)
        
    
    return ax.figure, ax


# In[30]:


results = more_plot_results_

fig, ax = more_plot(results)


# ## hists of outputs

# In[31]:


more_plot_results = m_day_results.groupby(['run']).mean()
hist_results = more_plot_results.loc[:, ['bed_utilisation', 'primary_bed_queue', 'revision_bed_queue',
                                        'primary_mean_los', 'revision_mean_los']] 

fig, ax = plt.subplots(5, 1, figsize=(8,12))
ax[0].hist(hist_results['bed_utilisation']);
ax[0].set_ylabel('bed utilisation')
ax[1].hist(hist_results['primary_bed_queue']);
ax[1].set_ylabel('primary_bed_queue');
ax[2].hist(hist_results['revision_bed_queue']);
ax[2].set_ylabel('revision_bed_queue');
ax[3].hist(hist_results['primary_mean_los']);
ax[3].set_ylabel('primary_bed_queue');
ax[4].hist(hist_results['revision_mean_los']);
ax[4].set_ylabel('revision_mean_los');



# In[ ]:





# In[ ]:





# In[ ]:




