# %%

import tensorflow as tf
from tensorflow import keras
import numpy as np
import joblib
import datetime as date
import pandas as pd

import data

# predictive models for temperature and PV generation

model_temperature = joblib.load('random_forest_model_temperature.pkl')
model_pv_generation = joblib.load('random_forest_model_pv_generation.pkl')

# data temperature and pv generation

data_real = pd.read_csv('Data_Madrid_2011-2016.csv')

# Convert the 'time' column to datetime
data_real['time'] = pd.to_datetime(data_real['time'], format='%Y%m%d:%H%M')

# Set the 'time' column as the index
data_real.set_index('time', inplace=True)

# Rename columns for clarity
data_real.columns = ['pv_generation', 'H_sun', 'temperature', 'WS10m', 'Int']

# Resample to 10-minute intervals using interpolation
data_real_10min = data_real.resample('10T').interpolate(method='linear')

# Function to add small noise
def add_noise(series, noise_level=0.01):
    noise = np.random.normal(0, noise_level, series.shape)
    return series + noise

def fix_series(series, threshold=0.05):
    series[np.abs(series) <= threshold] = 0
    return series

data_real_10min = add_noise(data_real_10min)
data_real_10min = fix_series(data_real_10min)

print(data_real_10min.head())
print(np.shape(data_real_10min))

temperature = {}
pv_generation = {}


startdate = date.datetime(2011, 1, 1, 0, 0)

i = 0
while i <= np.shape(data_real_10min)[0] - 24*6:
    temperature_day = []
    pv_generation_day = []
    for j in range(24*6):
        temperature_day.append(data_real_10min['temperature'][i])
        pv_generation_day.append(data_real_10min['pv_generation'][i])
        i += 1
    name = f'{startdate.year}-{startdate.month}-{startdate.day}'
    startdate += date.timedelta(days=1)

    temperature[name] = temperature_day
    pv_generation[name] = pv_generation_day

# %%

# Learning data
gamma = 0.99
learning_rate = 0.01
bigM = 1000000
DELTA_T = 1/60

# Data from data.py
def inTimeIntervals(time, intervals):
    index = 0
    for interval in intervals:
        if time >= interval[0] and time <= interval[1]:
            index = intervals.index(interval)

    return index

########## Shiftable ##########
operation_cycle_data = data.data_sm6["Operation Cycle"]
power_LW_data = data.data_sm6["Power_LW"]
power_LW_data = [0, power_LW_data[0], power_LW_data[1], power_LW_data[2], power_LW_data[3], power_LW_data[4], power_LW_data[5], 0, 0, 0, 0]
power_DW_data = data.data_sm6["Power_DW"]
power_DW_data = [0, power_DW_data[0], power_DW_data[1], power_DW_data[2], power_DW_data[3], power_DW_data[4], power_DW_data[5], power_DW_data[6], 0, 0, 0]
power_CD_data = data.data_sm6["Power_CD"]
power_CD_data = [0, power_CD_data[0], power_CD_data[1], power_CD_data[2], power_CD_data[3], power_CD_data[4], 0, 0, 0, 0, 0]
operation_length_DW_data = data.data_sm6["Operation Length DW"]
operation_length_LW_data = data.data_sm6["Operation Length LW"]
operation_length_CD_data = data.data_sm6["Operation Length CD"]

Tmin_DW = data.data_sm5["T_DW"][0]
Tmin_LW = data.data_sm5["T_LW"][0]
Tmin_CD = data.data_sm5["T_CD"][0]

Tmax_DW = data.data_sm5["T_DW"][1]
Tmax_LW = data.data_sm5["T_LW"][1]
Tmax_CD = data.data_sm5["T_CD"][1]

T_Shift = [[i for i in range(Tmin_DW, Tmax_DW)], [i for i in range(Tmin_LW, Tmax_LW)], [i for i in range(Tmin_CD, Tmax_CD)]]
Operation_Shift = [[1,2,3,4,5,6], [1,2,3,4,5,6,7], [1,2,3,4,5]]
power_shift = [power_DW_data, power_LW_data, power_CD_data]


########## EWH ##########
AU_data = data.data_sm7["AU"]
tau_net_data = data.data_sm7["tau_net"]
tau_min_data = data.data_sm7["tau_min"]
tau_max_data = data.data_sm7["tau_max"]
tau_req_data = data.data_sm7["tau_req"]
t_req_data = data.data_sm7["t_req"]
M_data = data.data_sm7["M"]
pr_data = data.data_sm7["pr"]
cp_data = data.data_sm7["cp"]

########## HVAC ##########
theta_min_data = data.data_sm10["theta_min"]
theta_max_data = data.data_sm10["theta_max"]
beta_data = data.data_sm10["beta"]
gamma_data = data.data_sm10["gamma"]
p_AC_nom_data = data.data_sm10["P_AC_nom"]

date_str = '2012-4-10'
theta_t_out_real = [temperature[date_str][i] for i in range(144) for _ in range(10)]
temperature_input_model = pd.DataFrame({'time': [i for i in range(1,144)], 'lag_1': temperature[date_str][:-1]})
theta_t_out_pred = model_temperature.predict(temperature_input_model)
theta_t_out_pred_shifted = np.insert(theta_t_out_pred, 0, theta_t_out_real[0])
theta_t_out_pred = [theta_t_out_pred_shifted[i] for i in range(144) for _ in range(10)]


########## EV ##########
eta_ch_B_data = data.data_sm12["eta_ch^B"]
eta_dch_B_data = data.data_sm12["eta_dch^B"]
E_min_B_data = data.data_sm12["E_min^B"]
E_max_B_data = data.data_sm12["E_max^B"]
P_ch_max_B_data = data.data_sm12["P_max^Bch"]
P_dch_max_B_data = data.data_sm12["P_max^Bdch"]

eta_ch_V_data = data.data_sm13["eta_ch^V"]
eta_dch_V_data = data.data_sm13["eta_dch^V"]
E_min_V_data = data.data_sm13["E_min^V"]
E_max_V_data = data.data_sm13["E_max^V"]
P_ch_max_V_data = data.data_sm13["P_max^Vch"]
P_dch_max_V_data = data.data_sm13["P_max^Vdch"]
ta_data = data.data_sm13["ta"]
td_data = data.data_sm13["td"]
E_B_req_data = data.data_sm12["E_0^B"]
E_V_req_data = data.data_sm13["E_req^V"]

pv_t_generation = pv_generation[date_str]
pv_t_generation_real = [pv_generation[date_str][i] for i in range(144) for _ in range(10)]
pv_generation_input_model = pd.DataFrame({'time': [i for i in range(1,144)], 'lag_1': pv_generation[date_str][:-1]})
pv_t_generation_pred = model_pv_generation.predict(pv_generation_input_model)
pv_t_generation_pred_shifted = np.insert(pv_t_generation_pred, 0, pv_t_generation_real[0])
pv_t_generation_pred = [pv_t_generation_pred_shifted[i] for i in range(144) for _ in range(10)]

Tb = [i for i in range(1440)]
Tv = [i for i in range(ta_data[0], td_data[0] + 1)]
Tx = [Tb, Tv]

########## Other ##########
P_G_max_data = data.data_sm14["Maximum Power for Grid Exchange"]   

########## Historical ##########

historical_s_LW = [0 for _ in range(1440)]
historical_s_DW = [0 for _ in range(1440)]
historical_s_CD = [0 for _ in range(1440)]

historical_n = [0 for _ in range(1440)]
historical_nu = [0 for _ in range(1440)]
historical_tau = [0 for _ in range(1440)]

historical_s_AC = [0 for _ in range(1440)]
historical_theta_in = [0 for _ in range(1440)]

historical_E_B = [1 for _ in range(1440)]
historical_E_V = [8 for _ in range(1440)]

historical_PG2H = [0 for _ in range(1440)]


optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = keras.losses.MeanSquaredError()

# %%

# state =       [0       , 1     , 2       , 3     , 4   , 5     , 6   , 7         , 8    , 9  , 10  , 11      , 12 , 13 , 14   , 15   , 16   , 17   , 18   , 19   , 20   , 21   , 22   , 23   , 24   , 25   , 26      , 27  ]
# state =   [s_DW_t_l, l_DW_t, s_LW_t_l, l_LW_t, s_CD_t_l, l_CD_t, nu_t, p_losses_t, tau_t, n_t, s_AC, theta_in, z_t, y_t, p_B2H, p_H2B, p_V2H, p_H2V, s_B2H, s_H2B, s_V2H, s_H2V, E_t_B, E_t_V, p_G2H, p_H2G, u_Cont_l, time]

def state_init(real_data):
    state = {

            ########## Shiftable ##########
            'shiftable_0_state':    0,
            'shiftable_0_length':   0,
            'shiftable_1_state':    0,
            'shiftable_1_length':   0,
            'shiftable_2_state':    0,
            'shiftable_2_length':   0,

            ########## EWH ##########
            'EWH_nu':               data.data_sm7["nu_0"][0],
            'EWH_p_losses':         data.data_sm7["p0_losses"][0],
            'EWH_tau':              data.data_sm7["tau_0"][0],
            'EWH_n':                0,

            ########## HVAC ##########
            'HVAC_state':           data.data_sm10["s0"][0],
            'HVAC_theta_in':        data.data_sm10["theta_in0"][0],
            'HVAC_theta_out':       real_data[0][0],
            'HVAC_z':               0,
            'HVAC_y':               0,

            ########## EV ##########
            'EV_pH2B':              0,
            'EV_pB2H':              0,
            'EV_pH2V':              0,
            'EV_pV2H':              0,
            'EV_sB2H':              0,
            'EV_sH2B':              0,
            'EV_sV2H':              0,
            'EV_sH2V':              0,
            'EV_E_t_B':             1,
            'EV_E_t_V':             8,
            'EV_PV_gen':            real_data[1][0],

            ########## Other ##########
            'P_G2H':                0,
            'timestep':             0
        }
    
    return state

# This function defines the action space for every timestep

# %%

def actionspace(state, prediction_data):

    timestep = state['timestep']

    ########## Shiftable ##########
    shiftable_actions = {
        'shiftable_0_state': [],
        'shiftable_0_length': [],
        'shiftable_1_state': [],
        'shiftable_1_length': [],
        'shiftable_2_state': [],
        'shiftable_2_length': []
    }

    for i in range(3):
        state_key = f'shiftable_{i}_state'
        length_key = f'shiftable_{i}_length'
        

        if timestep in T_Shift[i]:
            if state[state_key] == 0 and state[length_key] == 0:
                shiftable_actions[state_key] = [0, 1]
                shiftable_actions[length_key] = [0]

            elif state[length_key] == 14:
                if state[state_key] < Operation_Shift[i][-1]:
                    shiftable_actions[state_key] = [state[state_key] + 1]  # Machine goes to next state
                    shiftable_actions[length_key] = [0]
                else:
                    shiftable_actions[state_key] = [10]  # Machine turns off
                    shiftable_actions[length_key] = [0]

            elif state[state_key] == 10:
                shiftable_actions[state_key] = [10]  # Machine turns off
                shiftable_actions[length_key] = [0]

            else:
                shiftable_actions[state_key] = [state[state_key]]  # Continue operation
                shiftable_actions[length_key] = [state[length_key] + 1]
        
        else:
            shiftable_actions[state_key] = [state[state_key]]
            shiftable_actions[length_key] = [state[length_key]]

    ########## EWH ##########

    mt_data = data.data_sm8["Water Withdrawal"][inTimeIntervals(timestep, data.data_sm8["Time Interval"])]

    action = {

        ########## Shiftable ##########
        'shiftable_0_state':    shiftable_actions['shiftable_0_state'],
        'shiftable_0_length':   shiftable_actions['shiftable_0_length'],
        'shiftable_1_state':    shiftable_actions['shiftable_1_state'],
        'shiftable_1_length':   shiftable_actions['shiftable_1_length'],
        'shiftable_2_state':    shiftable_actions['shiftable_2_state'],
        'shiftable_2_length':   shiftable_actions['shiftable_2_length'],

        ########## EWH ##########
        'EWH_nu':               [0, 1],
        'EWH_p_losses':         [AU_data[0] * (state['EWH_tau'] - data.data_sm9["Ambient Temperature"][inTimeIntervals(timestep, data.data_sm9["Time Interval"])])],
        'EWH_tau':              [((M_data[0] - mt_data) / M_data[0]) * state['EWH_tau'] + (mt_data / M_data[0]) * tau_net_data[0] + ((pr_data[0] * state['EWH_nu'] - state['EWH_p_losses']) / (cp_data[0] * M_data[0])) * DELTA_T],
        'EWH_n':                [0, 1],

        ########## HVAC ##########
        'HVAC_state':           [0, 1],
        'HVAC_theta_out':       [prediction_data[0][timestep]],
        'HVAC_theta_in':        [beta_data[0] * (prediction_data[0][timestep] - state['HVAC_theta_in']) + gamma_data[0] * p_AC_nom_data[0] * state['HVAC_state']],
        'HVAC_z':               [0, 1],
        'HVAC_y':               [0, 1],

        ########## EV ##########
        'EV_pB2H':              [0, 1 / 4 * P_ch_max_B_data[0], 1 / 2 * P_ch_max_B_data[0], 3 / 4 * P_ch_max_B_data[0], P_ch_max_B_data[0]],
        'EV_pH2B':              [0, 1 / 4 * P_dch_max_B_data[0], 1 / 2 * P_dch_max_B_data[0], 3 / 4 * P_dch_max_B_data[0], P_dch_max_B_data[0]],
        'EV_pH2V':              [0, 1 / 4 * P_dch_max_V_data[0], 1 / 2 * P_dch_max_V_data[0], 3 / 4 * P_dch_max_V_data[0], P_dch_max_V_data[0]],
        'EV_pV2H':              [0, 1 / 4 * P_ch_max_V_data[0], 1 / 2 * P_ch_max_V_data[0], 3 / 4 * P_ch_max_V_data[0], P_ch_max_V_data[0]],
        'EV_sB2H':              [0, 1],
        'EV_sH2B':              [0, 1],
        'EV_sV2H':              [0, 1],
        'EV_sH2V':              [0, 1],
        'EV_E_t_B':             [(eta_ch_B_data[0] * state['EV_pH2B'] * state['EV_sH2B'] - eta_dch_B_data[0] * state['EV_pB2H'] - state['EV_sB2H']) * DELTA_T],
        'EV_E_t_V':             [(eta_ch_V_data[0] * state['EV_pH2V'] * state['EV_sH2V'] - eta_dch_V_data[0] * state['EV_pV2H'] - state['EV_sV2H']) * DELTA_T],
        'EV_PV_gen':            [prediction_data[1][timestep]],

        ########## Other ##########
        'P_G2H':                [data.data_sm3['Power'][inTimeIntervals(timestep, data.data_sm3['Time Interval'])] 
                                 - prediction_data[1][timestep]
                                 + p_AC_nom_data[0] * state['HVAC_state'] 
                                 + pr_data[0] * state['EWH_nu'] 
                                 + state['EV_pH2B'] 
                                 - state['EV_pB2H'] 
                                 + int(timestep in Tx[1]) * (state['EV_pH2V'] - state['EV_pV2H'])
                                 + power_shift[0][state['shiftable_0_state']] 
                                 + power_shift[1][state['shiftable_1_state']] 
                                 + power_shift[2][state['shiftable_2_state']]],
        'timestep':             [1]
    }

    return action


def post_decision_state_transition(state, action):

    post_decision_state = state.copy()

    timestep = state['timestep']

    ########## Shiftable ##########

    post_decision_state['shiftable_0_state'] =  action['shiftable_0_state']
    post_decision_state['shiftable_0_length'] = action['shiftable_0_length']
    post_decision_state['shiftable_1_state'] =  action['shiftable_1_state']
    post_decision_state['shiftable_1_length'] = action['shiftable_1_length']
    post_decision_state['shiftable_2_state'] =  action['shiftable_2_state']
    post_decision_state['shiftable_2_length'] = action['shiftable_2_length']
     
    ########## EWH ##########

    post_decision_state['EWH_nu'] =             action['EWH_nu']
    post_decision_state['EWH_p_losses'] =       action['EWH_p_losses']
    post_decision_state['EWH_tau'] =            action['EWH_tau']
    post_decision_state['EWH_n'] =              action['EWH_n']

    ########## HVAC ##########

    post_decision_state['HVAC_state'] =         action['HVAC_state']
    post_decision_state['HVAC_theta_out'] =     action['HVAC_theta_out']
    post_decision_state['HVAC_theta_in'] =      state['HVAC_theta_in'] + action['HVAC_theta_in']
    post_decision_state['HVAC_z'] =             action['HVAC_z']
    post_decision_state['HVAC_y'] =             action['HVAC_y']

    ########## EV ##########

    post_decision_state['EV_pB2H'] =            action['EV_pB2H']
    post_decision_state['EV_pH2B'] =            action['EV_pH2B']
    post_decision_state['EV_pH2V'] =            action['EV_pH2V']
    post_decision_state['EV_pV2H'] =            action['EV_pV2H']
    post_decision_state['EV_sB2H'] =            action['EV_sB2H']
    post_decision_state['EV_sH2B'] =            action['EV_sH2B']
    post_decision_state['EV_sV2H'] =            action['EV_sV2H']
    post_decision_state['EV_sH2V'] =            action['EV_sH2V']

    post_decision_state['EV_E_t_B'] =           state['EV_E_t_B'] + action['EV_E_t_B']
    post_decision_state['EV_E_t_V'] =           state['EV_E_t_V'] + action['EV_E_t_V']
    post_decision_state['EV_PV_gen'] =          action['EV_PV_gen']

    ########## Other ##########

    post_decision_state['P_G2H'] =              action['P_G2H']
    post_decision_state['timestep'] =           state['timestep'] + action['timestep']

    ########## Historical ##########
    historical_s_LW[timestep + 1] =             post_decision_state['shiftable_0_state']
    historical_s_DW[timestep + 1] =             post_decision_state['shiftable_1_state']
    historical_s_CD[timestep + 1] =             post_decision_state['shiftable_2_state']

    historical_n[timestep + 1] =                post_decision_state['EWH_n']
    historical_nu[timestep + 1] =               post_decision_state['EWH_nu']
    historical_tau[timestep + 1] =              post_decision_state['EWH_tau']

    historical_s_AC[timestep + 1] =             post_decision_state['HVAC_state']
    historical_theta_in[timestep + 1] =         post_decision_state['HVAC_theta_in']

    historical_E_B[timestep + 1] =              post_decision_state['EV_E_t_B']
    historical_E_V[timestep + 1] =              post_decision_state['EV_E_t_V']

    historical_PG2H[timestep + 1] =             post_decision_state['P_G2H']

    return post_decision_state


def next_pre_decision_state(post_decision_state, real_data, prediction_data):

    new_state = post_decision_state.copy()
    timestep = new_state['timestep']

    new_state['HVAC_theta_out'] = real_data[0][timestep]
    new_state['HVAC_theta_in'] = post_decision_state['HVAC_theta_in'] + beta_data[0] * (real_data[0][timestep] - prediction_data[0][timestep])

    new_state['EV_PV_gen'] = real_data[1][timestep]
    new_state['P_G2H'] = post_decision_state['P_G2H'] + new_state['EV_PV_gen'] - prediction_data[1][timestep]

    return new_state

        
def reward(state):

    timestep = state["timestep"]

    ###############################
    #### Constraints Violation ####
    ###############################

    constraint_count = 0

    ########## Shiftable ##########

    # C 9 - Shiftable machines must have run at the end of their cycle
    for i in range(3):
        if timestep >= T_Shift[i][-1] - Operation_Shift[i][-1] * 15 and state[f'shiftable_{i}_state'] == 0:
            constraint_count += 1/(timestep-T_Shift[i][-1])

    ########## EWH ##########

    # C 21 
    if state['EWH_tau'] < tau_min_data[0] - bigM * state['EWH_nu']:
        constraint_count += 1

    # C 22
    if state['EWH_tau'] > tau_max_data[0] + bigM * (1 - state['EWH_nu']):
        constraint_count += 1

    # C 23
    if timestep == 1439 - tau_req_data[0] + 1:
        sum = 0
        for i in range(1439-tau_req_data[0]+1):
            sum += state['EWH_n']
        
        if sum != 1:
            constraint_count += 10

    # C 24
    sum_expr = 0
    for ts in range(tau_req_data[0]):
        if ts <= timestep:
            sum_expr += tau_req_data[0] * historical_n[timestep - ts]
    if state['EWH_tau'] < sum_expr:
        constraint_count += 1

    ########## HVAC ##########

    # C 33
    if state['HVAC_theta_in'] - bigM * state['HVAC_state'] < theta_min_data[0]:
        constraint_count += 1

    # C 34
    if state['HVAC_theta_in'] + bigM * state['HVAC_z'] > theta_min_data[0]:
        constraint_count += 1

    # C 35
    if state['HVAC_theta_in'] - bigM * state['HVAC_y'] < theta_max_data[0]:
        constraint_count += 1

    # C 36
    if historical_s_AC[timestep] - historical_s_AC[timestep-1] + state['HVAC_z'] + state['HVAC_y'] > 2:
        constraint_count += 1

    # C 37
    if historical_s_AC[timestep-1] - historical_s_AC[timestep] + state['HVAC_z'] + state['HVAC_y'] > 2:
        constraint_count += 1

    # C 38
    if state['HVAC_theta_in'] > theta_max_data[0] - bigM * (1-state['HVAC_state']):
        constraint_count += 1

    ########## EV ##########

    # C 48
    if E_min_B_data[0] > state['EV_E_t_B']:
        constraint_count += 1

    if E_max_B_data[0] < state['EV_E_t_B']:
        constraint_count += 1

    if E_min_V_data[0] > state['EV_E_t_V']:
        constraint_count += 1

    if E_max_V_data[0] < state['EV_E_t_V']:
        constraint_count += 1

    # C 49
    if state['EV_pH2B'] > P_ch_max_B_data[0] * state['EV_sH2B']:
        constraint_count += 1

    if state['EV_pB2H'] > P_dch_max_B_data[0] * state['EV_sB2H']:
        constraint_count += 1

    # C 50
    if state['EV_pH2V'] > P_ch_max_V_data[0] * state['EV_sH2V']:
        constraint_count += 1

    if state['EV_pV2H'] > P_dch_max_V_data[0] * state['EV_sV2H']:
        constraint_count += 1

    if timestep not in Tx[1] and state['EV_pH2V'] > 0:
        constraint_count += 1


    # C 51
    if state['EV_pH2B'] + state['EV_sB2H'] > 1:
        constraint_count += 1

    if state['EV_pH2V'] + state['EV_sV2H'] > 1:
        constraint_count += 1

    # C 52
    if timestep == 1438:
        if state['EV_E_t_B'] < E_B_req_data[0]:
            constraint_count += 100

    if timestep == td_data[0]:
        if state['EV_E_t_V'] < E_V_req_data[0]:
            constraint_count += 100

    ########## Other ##########

    # C 56

    if state['P_G2H'] > P_G_max_data[0] or state['P_G2H'] < -P_G_max_data[0]:
        constraint_count += 1


    ###############################
    ###### Objective Function #####
    ###############################

    cbuy = data.data_sm1["Electricity Price"][inTimeIntervals(timestep, data.data_sm1["Time Interval"])]
    # Assuming that cbuy = csell

    costs = state['P_G2H'] * cbuy * DELTA_T
    constraint_violation_penalty = 1000 * constraint_count

    reward = costs + constraint_violation_penalty
    print(f"reward: {reward}")

    return reward


def model(n, m, input_size, output_size):
    model = keras.Sequential()
    model.add(keras.layers.Dense(m, activation='relu', input_shape=(input_size,)))

    for _ in range(n-1):
        model.add(keras.layers.Dense(m, activation='relu'))

    model.add(keras.layers.Dense(output_size, activation='softmax'))
    return model

# %%
import random
import itertools

def select_action(state, model, prediction_data,  epsilon=0.1,):
    if random.random() < epsilon:
        action = actionspace(state, prediction_data)
        action_choice = {key: random.choice(value) for key, value in action.items()}
    else:
        state_input = np.expand_dims(state_to_array(state), axis=0)
        action = actionspace(state, prediction_data)
        action_values = model.predict(state_input)
        action_index = np.argmax(action_values)
        action = index_to_action(action_index, action)  # Convert index to action
        action_choice = {key: action[key] for key in action.keys()}
    return action_choice

# Flatten the action space to create a list of all possible actions
def flatten_action_space(action_space):
    keys = sorted(action_space.keys())
    values = [action_space[key] for key in keys]
    all_combinations = list(itertools.product(*values))
    return all_combinations

# Convert index to action dictionary
def index_to_action(index, action_space):
    all_combinations = flatten_action_space(action_space)
    selected_action = all_combinations[index]
    action_dict = {key: selected_action[i] for i, key in enumerate(sorted(action_space.keys()))}
    return action_dict

def state_to_array(state):
    # Convert the state dictionary to a numpy array for model input
    state_array = np.array(list(state.values()))
    return state_array
# %% 
# Example usage
state_example = state_init([theta_t_out_real, pv_t_generation_real])  # Example state, needs to be properly defined
action_space_example = actionspace(state_example, [theta_t_out_pred, pv_t_generation_pred])
index = 0  # Example index
action_dict = index_to_action(index, action_space_example)

def check_for_nans(data, name):
    if np.isnan(data).any():
        print(f"NaN found in {name}")
    if np.isinf(data).any():
        print(f"Inf found in {name}")


def TD_learning(model, state_0, real_data, prediction_data, epochs=100, epsilon=0.1):
    for epoch in range(epochs):
        state = state_0
        for t in range(0, 1439):
            # Select action
            action = select_action(state, model, prediction_data, epsilon)
            
            # Transition to post-decision state
            post_decision_state = post_decision_state_transition(state, action)
            
            # Compute the reward
            reward_value = reward(state)
            check_for_nans(reward_value, "reward_value")
            
            # Transition to next state
            next_state = next_pre_decision_state(post_decision_state, real_data, prediction_data)
            next_real = np.expand_dims(state_to_array(next_state), axis=0)

            # Step 1: Prediction for current state
            state_input = np.expand_dims(state_to_array(state), axis=0)
            check_for_nans(state_input, "state_input")
            prediction = model.predict(state_input)

            # Step 2: Prediction for next state
            next_state_input = np.expand_dims(state_to_array(next_state), axis=0)
            check_for_nans(next_state_input, "next_state_input")
            next_prediction = model.predict(next_state_input)

            # Step 3: Compute TD target
            check_for_nans(next_prediction, "next_prediction")

            # print(f'next_state_input: {next_state_input}')

            td_target = reward_value + gamma * next_prediction
            check_for_nans(td_target, "td_target")

            # Step 4: Compute loss and gradients
            with tf.GradientTape() as tape:
                current_prediction = model(state_input, training=True)
                check_for_nans(current_prediction, "current_prediction")
                loss = loss_fn(td_target, current_prediction)
                check_for_nans(loss, "loss")

            # Step 5: Calculate gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            for idx, grad in enumerate(gradients):
                check_for_nans(grad, f"gradient_{idx}")

            # Step 6: Apply gradients to update model parameters with gradient clipping
            clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
            optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))

            # Update state
            state = next_state

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}")

# %%

########################
##### MAIN PROGRAM #####
########################

# Define the model
n = 2
m = 32 
input_size = 28
output_size = 28
model = model(n, m, input_size, output_size)

# Define the data
date_str = '2012-4-10'
theta_t_out_real = [temperature[date_str][i] for i in range(144) for _ in range(10)]
temperature_input_model = pd.DataFrame({'time': [i for i in range(1,144)], 'lag_1': temperature[date_str][:-1]})
theta_t_out_pred = model_temperature.predict(temperature_input_model)
theta_t_out_pred_shifted = np.insert(theta_t_out_pred, 0, theta_t_out_real[0])
theta_t_out_pred = [theta_t_out_pred_shifted[i] for i in range(144) for _ in range(10)]

pv_t_generation = pv_generation[date_str]
pv_t_generation_real = [pv_generation[date_str][i] for i in range(144) for _ in range(10)]
pv_generation_input_model = pd.DataFrame({'time': [i for i in range(1,144)], 'lag_1': pv_generation[date_str][:-1]})
pv_t_generation_pred = model_pv_generation.predict(pv_generation_input_model)
pv_t_generation_pred_shifted = np.insert(pv_t_generation_pred, 0, pv_t_generation_real[0])
pv_t_generation_pred = [pv_t_generation_pred_shifted[i] for i in range(144) for _ in range(10)]

real_data = [theta_t_out_real, pv_t_generation_real]
prediction_data = [theta_t_out_pred, pv_t_generation_pred]

state_0 = state_init(real_data)
# %% 

# Train the model
TD_learning(model, state_0, real_data, prediction_data, epochs=100, epsilon=0.1)

# %%
joblib.dump(model, 'model.pkl')