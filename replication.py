import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import data
import numpy as np
from matplotlib import pyplot as plt

def inTimeIntervals(time, intervals):
    index = 0
    for interval in intervals:
        if time >= interval[0] and time <= interval[1]:
            index = intervals.index(interval)

    return index


DELTA_T = 1/60
BIGM = 10000

# Define which model to run

OVERALL = True
SHIFTABLE = False
HVAC = False
EWH = False
SATISFACTION = False

# Initialize the model
m = gp.Model('Replication Model')

# Restriction variables to sum up binary variables

# Assuming Tmin values are defined in data or variables
Tmin_DW = data.data_sm5["T_DW"][0]
Tmin_LW = data.data_sm5["T_LW"][0]
Tmin_CD = data.data_sm5["T_CD"][0]

Tmin = [Tmin_DW, Tmin_LW, Tmin_CD]

Tmax_DW = data.data_sm5["T_DW"][1]
Tmax_LW = data.data_sm5["T_LW"][1]
Tmax_CD = data.data_sm5["T_CD"][1]

Tmax = [Tmax_DW, Tmax_LW, Tmax_CD]

power_shift = [data.data_sm6["Power_DW"], data.data_sm6["Power_LW"], data.data_sm6["Power_CD"]]

d_shift = [data.data_sm6["Operation Length DW"][0], data.data_sm6["Operation Length LW"][0], data.data_sm6["Operation Length CD"][0]]

D_shift = []

# Loop through the allowed time slots, considering the duration constraint
operation_cycle = data.data_sm6["Operation Cycle"]
for i in range(3):
    matrix = []
    for t in range(Tmin[i], Tmax[i] - d_shift[i] + 1):
        row = []
        for ts in range(Tmin[i], Tmax[i]):
            value = 0
            if t in range(ts - d_shift[i] + 1, ts + 1):
                interval_index = inTimeIntervals(ts - t + 1, operation_cycle)
                if interval_index is not None:
                    value = power_shift[i][interval_index]
            row.append(value)
        matrix.append(row)
    D_shift.append(matrix)

Tb = [i for i in range(1440)]
Tv = [i for i in range(data.data_sm13["ta"][0], data.data_sm13["td"][0] + 1)]
Tx = [Tb, Tv]

if OVERALL or SATISFACTION:
    # Overall model

    u_Cont = [m.addVar(vtype=GRB.BINARY, name=f'u_Cont_{i}') for i in range(9)]

    p_G2H = [m.addVar(vtype=GRB.CONTINUOUS, name=f'p_G2H_{t}', lb=0) for t in range(1440)]
    p_H2G = [m.addVar(vtype=GRB.CONTINUOUS, name=f'p_H2G_{t}', lb=0) for t in range(1440)]

    sH2G = [m.addVar(vtype=GRB.BINARY, name=f'sH2G_{t}') for t in range(1440)]
    sG2H = [m.addVar(vtype=GRB.BINARY, name=f'sG2H_{t}') for t in range(1440)]

# EV

    p_x2H = [[None for _ in range(2)] for _ in range(1440)]
    p_H2x = [[ None for _ in range(2)] for _ in range(1440)]

    E_tx = [[ None for _ in range(2)] for _ in range(1440)]
    s_H2x = [[ None for _ in range(2)] for _ in range(1440)]
    s_x2H = [[ None for _ in range(2)] for _ in range(1440)]

    # Populate only the time periods in Tx
    for x in range(2):
        for t in Tx[x]:
            p_x2H[t][x] = m.addVar(vtype=GRB.CONTINUOUS, name=f'p_x2H_{t}_{x}', lb=0)
            p_H2x[t][x] = m.addVar(vtype=GRB.CONTINUOUS, name=f'p_H2x_{t}_{x}', lb=0)

            E_tx[t][x] = m.addVar(vtype=GRB.CONTINUOUS, name=f'E_tx_{t}_{x}', lb=0)
            s_H2x[t][x] = m.addVar(vtype=GRB.BINARY, name=f's_H2x_{t}_{x}')
            s_x2H[t][x] = m.addVar(vtype=GRB.BINARY, name=f's_x2H_{t}_{x}')

    # p_x2H = [[m.addVar(vtype=GRB.CONTINUOUS, name=f'p_x2H_{t}_{i}', lb=0) for i in range(2)] for t in range(1440)]
    # p_H2x = [[m.addVar(vtype=GRB.CONTINUOUS, name=f'p_H2x_{t}_{i}', lb=0) for i in range(2)] for t in range(1440)]

    # E_tx = [[m.addVar(vtype=GRB.CONTINUOUS, name=f'E_tx_{t}_{i}', lb=0) for i in range(2)] for t in range(1440)]
    # s_H2x = [[m.addVar(vtype=GRB.BINARY, name=f's_H2x_{t}_{i}') for i in range(2)] for t in range(1440)]
    # s_x2H = [[m.addVar(vtype=GRB.BINARY, name=f's_x2H_{t}_{i}') for i in range(2)] for t in range(1440)]

if SHIFTABLE or OVERALL or SATISFACTION:
    # Shiftable load

    s_shift = [[m.addVar(vtype=GRB.BINARY, name=f's_shift_{t}_{i}') for i in range(3)] for t in range(1440)]
    p_shift = [[m.addVar(vtype=GRB.CONTINUOUS, name=f'p_shift_{t}_{i}', lb=0) for i in range(3)] for t in range(1440)]


if HVAC or OVERALL or SATISFACTION:
    # HVAC
    sAC = [m.addVar(vtype=GRB.BINARY, name=f'sAC_{t}') for t in range(1440)]
    theta = [m.addVar(vtype=GRB.CONTINUOUS, name=f'theta_{t}') for t in range(1440)]
    y = [m.addVar(vtype=GRB.BINARY, name=f'y_{t}') for t in range(1440)]
    z = [m.addVar(vtype=GRB.BINARY, name=f'z_{t}') for t in range(1440)]

if EWH or OVERALL or SATISFACTION:
    # EWH
    nu = [m.addVar(vtype=GRB.BINARY, name=f'nu_{t}') for t in range(1440)]
    tau = [m.addVar(vtype=GRB.CONTINUOUS, name=f'tau_{t}') for t in range(1440)]
    n = [m.addVar(vtype=GRB.BINARY, name=f'n_{t}') for t in range(1440)]
    pLosses = [m.addVar(vtype=GRB.CONTINUOUS, name=f'pLosses_{t}', lb=0) for t in range(1440)]


# Create variables and constraints
for t in range(0, 1440):
    ##############################
    #####    Restrictions    #####
    ##############################

    # Overall model restrictions

    if OVERALL or SATISFACTION:
        # C 54

        sum = 0
        for i in range(9):
            sum += data.data_sm2["Max Power"][i] * u_Cont[i]
        
        m.addConstr(p_G2H[t] <= sum, name = f'C54_{t}')

        # C 59

        m.addConstr(p_G2H[t] <= data.data_sm14["Maximum Power for Grid Exchange"][0] * sG2H[t], name = f'C59_{t}')

        # C 60

        m.addConstr(p_H2G[t] <= data.data_sm14["Maximum Power for Grid Exchange"][0] * sH2G[t], name = f'C60_{t}')

        # C 61

        m.addConstr(sG2H[t] + sH2G[t] <= 1, name = f'C61_{t}')

        # C 62 + 63
        
        Bt = data.data_sm3["Power"][inTimeIntervals(t, data.data_sm3["Time Interval"])]
        P_pvt = data.data_sm4["PV Energy Generation"][inTimeIntervals(t, data.data_sm4["Time Interval"])]
        P_ac = data.data_sm10["P_AC_nom"][0]
        P_r = data.data_sm7["pr"][0]

        p_shift_total = 0
        for i in range(3):
            p_shift_total += p_shift[t][i]

        if t in Tx[1]:
            m.addConstr(p_G2H[t] - p_H2G[t] + P_pvt == Bt + p_shift_total + P_ac * sAC[t] + P_r * nu[t] + (p_H2x[t][0] - p_x2H[t][0]) + (p_H2x[t][1] - p_x2H[t][1]), name=f'C63_{t}')
        else:
            m.addConstr(p_G2H[t] - p_H2G[t] + P_pvt == Bt + p_shift_total + P_ac * sAC[t] + P_r * nu[t] + (p_H2x[t][0] - p_x2H[t][0]), name=f'C62_{t}')

    if SHIFTABLE or OVERALL or SATISFACTION:
        # Shiftable load restrictions

        # C 8+10

        for i in range(3):
            sum_expr = gp.LinExpr()
            if Tmin[i] <= t <= Tmax[i]:
                for ts in range(Tmin[i], Tmax[i] - d_shift[i] + 1):

                    # print(f'i={i}, t={t}, ts={ts}, ts-Tmin[i]={ts-Tmin[i]}, t-Tmin[i]={t-Tmin[i]}')

                    if (0 <= ts-Tmin[i] < len(D_shift[i])) and (0 <= t-Tmin[i] < len(D_shift[i][0])):
                        sum_expr += D_shift[i][ts-Tmin[i]][t-Tmin[i]] * s_shift[ts][i]
                    # else:
                        # print(f'Index out of range: i={i}, ts={ts-Tmin[i]}, t={t-Tmin[i]}')

                m.addConstr(p_shift[t][i] == sum_expr, name=f'C8_{t}_{i}')
            else:
                m.addConstr(p_shift[t][i] == 0, name=f'C10_{t}_{i}')


    # HVAC restrictions

    if HVAC or OVERALL or SATISFACTION:

        # C 32
        beta = data.data_sm10["beta"][0]
        gamma = data.data_sm10["gamma"][0]
        pAC = data.data_sm10["P_AC_nom"][0]
        if t > 0:
            m.addConstr(theta[t] == (1-beta) * theta[t-1] + beta * data.data_sm11["theta_t_ext"][inTimeIntervals(t-1, data.data_sm11["Time Interval"])] + gamma * sAC[t-1] * pAC, name=f'C32_{t}')    
        else:
            m.addConstr(theta[t] == (1-beta) * data.data_sm10["theta_in0"][0] + beta * data.data_sm11["theta_t_ext"][inTimeIntervals(t, data.data_sm11["Time Interval"])], name=f'C32_{t}')
            
        # C 33
        m.addConstr(theta[t] >= data.data_sm10["theta_min"][0] - BIGM * sAC[t], name=f'C33_{t}')

        # C 34
        m.addConstr(theta[t] <= data.data_sm10["theta_min"][0] + BIGM * z[t], name=f'C34_{t}')

        # C 35
        m.addConstr(theta[t] >= data.data_sm10["theta_max"][0] - BIGM * y[t], name=f'C35_{t}')

        # C 36
        if t > 0:
            m.addConstr(y[t] + z[t] + sAC[t] - sAC[t-1] <= 2, name=f'C36_{t}')
        
        else:
            m.addConstr(y[t] + z[t] + sAC[t] <= 2, name=f'C36_{t}')

        # C 37
        if t > 0:
            m.addConstr(y[t] + z[t] - sAC[t] + sAC[t-1] <= 2, name=f'C37_{t}')
        
        else:
            m.addConstr(y[t] + z[t] - sAC[t] <= 2, name=f'C37_{t}')

        # C 38
        m.addConstr(theta[t] <= data.data_sm10["theta_max"][0] + BIGM * (1-sAC[t]), name=f'C38_{t}')


    if EWH or OVERALL or SATISFACTION:

        # EWH restrictions

        # C 19
        tau_amb_t = data.data_sm9["Ambient Temperature"][inTimeIntervals(t, data.data_sm9["Time Interval"])]
        m.addConstr(pLosses[t] == data.data_sm7["AU"][0] * (tau[t] - tau_amb_t), name=f'C19_{t}')


        # C 20
        mt = data.data_sm8["Water Withdrawal"][inTimeIntervals(t-1, data.data_sm8["Time Interval"])]
        M = data.data_sm7["M"][0]
        Pr = data.data_sm7["pr"][0]
        cp = data.data_sm7["cp"][0]
        tau_net = data.data_sm7["tau_net"][0]

        if t > 0:
            m.addConstr(tau[t] == ((M - mt)/M)*tau[t-1] + (mt/M)*tau_net + ((1000 * Pr * nu[t-1] - pLosses[t-1])/(M * cp))*DELTA_T, name=f'C20_{t}')
        else:
            m.addConstr(tau[0] == data.data_sm7["tau_0"][0] + (Pr * data.data_sm7["nu_0"][0] - data.data_sm7["p0_losses"][0])/(M * cp)*DELTA_T, name=f'C20_{t}')

        # C 21
        m.addConstr(tau[t] >= data.data_sm7["tau_min"][0] - BIGM * nu[t], name=f'C21_{t}')

        # C 22
        m.addConstr(tau[t] <= data.data_sm7["tau_max"][0] + BIGM * (1-nu[t]), name=f'C22_{t}')

        # C 24
        t_req = data.data_sm7["t_req"][0]
        tau_req = data.data_sm7['tau_req'][0]

        sum_expr = gp.quicksum(tau_req * n[t - ts] for ts in range(t_req) if ts <= t)
        m.addConstr(tau[t] >= sum_expr, name=f'C24_{t}')

        # Add constraints regarding the discomfort of the consumer

    
    if OVERALL or SATISFACTION:

        # EV constraints

        # C 47
        eta_ch = [data.data_sm12["eta_ch^B"][0], data.data_sm13["eta_ch^V"][0]]
        eta_dch = [data.data_sm12["eta_dch^B"][0], data.data_sm13["eta_dch^V"][0]]
        E_0x = [data.data_sm12["E_0^B"][0], data.data_sm13["E_ta^V"][0]]

        for i in range(2):
            if t in Tx[i]:
                if i == 1 and (t == Tx[i][0]):
                    m.addConstr(E_tx[t][i] == E_0x[i], name=f'C47_{t}_{i}')

                elif i == 0 and t == 0:
                    m.addConstr(E_tx[t][i] == E_0x[i], name=f'C47_{t}_{i}')

                else: 
                    m.addConstr(E_tx[t][i] == E_tx[t-1][i] + eta_ch[i] * p_H2x[t][i] * DELTA_T - p_x2H[t][i] * DELTA_T / eta_dch[i], name=f'C47_{t}_{i}')


        # C 48
        E_min = [data.data_sm12["E_min^B"][0], data.data_sm13["E_min^V"][0]]
        E_max = [data.data_sm12["E_max^B"][0], data.data_sm13["E_max^V"][0]]

        for i in range(2):
            if t in Tx[i]:
                m.addConstr(E_tx[t][i] >= E_min[i], name=f'C48_{t}_{i}')
                m.addConstr(E_tx[t][i] <= E_max[i], name=f'C48_{t}_{i}')


        # C 49
        P_ch_max = [data.data_sm12["P_max^Bch"][0], data.data_sm13["P_max^Vch"][0]]

        for i in range(2):
            if t in Tx[i]:
                m.addConstr(p_H2x[t][i] <= P_ch_max[i] * s_H2x[t][i], name=f'C49_{t}_{i}') 

        # C 50
        P_dch_max = [data.data_sm12["P_max^Bdch"][0], data.data_sm13["P_max^Vdch"][0]]

        for i in range(2):
            if t in Tx[i]:
                m.addConstr(p_x2H[t][i] <= P_dch_max[i] * s_x2H[t][i], name=f'C50_{t}_{i}')

        # C 51
        for i in range(2):
            if t in Tx[i]:
                m.addConstr(s_H2x[t][i] + s_x2H[t][i] <= 1, name=f'C51_{t}_{i}')

        # C extra
        if t == 465 and i == 1:
            m.addConstr(p_x2H[t][1] == 0)



# Add constraints outside the loop if they involve summation over all time periods

##############################
#####    Restrictions    #####
##############################

# Overall model restrictions

if OVERALL or SATISFACTION:

    # C 52

    m.addConstr(E_tx[1439][0] >= data.data_sm12["E_0^B"][0], name = f'C52_{0}')
    m.addConstr(E_tx[data.data_sm13["td"][0]][1] >= data.data_sm13["E_req^V"][0], name=f'C52_{1}')

    # C 55

    sum = 0
    for i in range(9):
        sum += u_Cont[i]

    m.addConstr(sum == 1, name = f'C55')

# Shiftable load restrictions

if SHIFTABLE or OVERALL or SATISFACTION:

    # C 9
    for i in range(3):
        sum_expr = gp.LinExpr()
        for t in range(Tmin[i], Tmax[i] - d_shift[i]):
            sum_expr += s_shift[t][i]
        m.addConstr(sum_expr == 1, name=f'C9_{i}')

# HVAC restrictions
# Placeholder for HVAC constraints

# EWH restrictions

if EWH or OVERALL or SATISFACTION:

    # C 23
    sum = 0 
    for t in range(0, 1440 - data.data_sm7["tau_req"][0]):
        sum += n[t]

    m.addConstr(sum == 1, name='C23')

    # EV restrictions
    # Placeholder for EV constraints

# Set the objective function
objective_expr = gp.LinExpr()

if SHIFTABLE:
    for t in range(1440):
        interval_index = inTimeIntervals(t, data.data_sm1["Time Interval"])
        c_buy = data.data_sm1["Electricity Price"][interval_index]
        objective_expr += (c_buy * (p_shift[t][0] + p_shift[t][1] + p_shift[t][2])) * DELTA_T

if EWH:
    for t in range(1440):
        interval_index = inTimeIntervals(t, data.data_sm1["Time Interval"])
        c_buy = data.data_sm1["Electricity Price"][interval_index]
        objective_expr += c_buy * data.data_sm7["pr"][0] * nu[t] * DELTA_T

if HVAC:
    for t in range(1440):
        interval_index = inTimeIntervals(t, data.data_sm1["Time Interval"])
        c_buy = data.data_sm1["Electricity Price"][interval_index]
        objective_expr += c_buy * data.data_sm10["P_AC_nom"][0] * sAC[t] * DELTA_T

if OVERALL:
    # Calculate the sum for the objective function
    for t in range(1440):
        interval_index = inTimeIntervals(t, data.data_sm1["Time Interval"])
        c_buy = data.data_sm1["Electricity Price"][interval_index]
        c_sell = 0.52*c_buy  # This value works for the replication

        objective_expr += (c_buy * p_G2H[t] - c_sell * p_H2G[t]) * DELTA_T

    for l in range(len(data.data_sm2["Power Level"])):
        objective_expr += data.data_sm2["Price"][l] * u_Cont[l]

if SATISFACTION:
    # Calculate the sum for the objective function
    for t in range(1440):
        interval_index = inTimeIntervals(t, data.data_sm1["Time Interval"])
        c_buy = data.data_sm1["Electricity Price"][interval_index]
        c_sell = 0.53*c_buy  

        objective_expr += (c_buy * p_G2H[t] - c_sell * p_H2G[t]) * DELTA_T

    
# Objective function for the overall model
m.setObjective(objective_expr, GRB.MINIMIZE)

# Optimize the model
m.setParam(GRB.Param.TimeLimit, 10*60)

# Optimize the model
m.optimize()

# Print solution
if m.status == GRB.INFEASIBLE or m.status == GRB.INF_OR_UNBD:
    m.computeIIS()
    m.write("model.ilp")

    # Read the IIS file to find the infeasible constraints
    infeasible_constraints = []
    with open("model.ilp", "r") as f:
        capture = False
        for line in f:
            line = line.strip()
            if line.startswith("Subject To"):
                capture = True
            elif line.startswith("Bounds") or line.startswith("End"):
                capture = False
            elif capture and line.startswith("C"):
                constraint_name = line.split(":")[0].strip()
                infeasible_constraints.append(constraint_name)

    # Write infeasible constraints to a text file
    with open("infeasible_constraints.txt", "w") as f:
        f.write("Infeasible constraints:\n")
        for constraint in infeasible_constraints:
            f.write(constraint + "\n")

    print("Infeasible constraints written to infeasible_constraints.txt")

    # After defining and optimizing the model

    
else:
    print("Optimal solution found.")
    print(f"Status: {m.status}")
    print(f"Objective value: {m.ObjVal}")

    if SHIFTABLE:
        with open("feasible_shift.txt", "w") as f:
            for t in range(1440):
                for i in range(3):
                    f.write(f"s_shift_{t}_{i} = {s_shift[t][i].X}\n")
                    f.write(f"p_shift_{t}_{i} = {p_shift[t][i].X}\n\n")

        output_P_DW = [p_shift[t][0].X for t in range(1440)]
        output_P_LW = [p_shift[t][1].X for t in range(1440)]
        output_P_CD = [p_shift[t][2].X for t in range(1440)]

        # Create the plot
        plt.figure(figsize=(10, 5))

        plt.plot(range(1440), output_P_DW, color='blue', label='PDW')
        plt.plot(range(1440), output_P_LW, color='orange', label='PLW')
        plt.plot(range(1440), output_P_CD, color='green', label='PCD')

        # Add labels and title
        plt.xlabel('Time (minutes)')
        plt.ylabel('Power')
        plt.title('Power over Time')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()

    if EWH:
        with open("feasible_ewh.txt", "w") as f:
            for t in range(1440):
                f.write(f"nu_{t} = {nu[t].X}\n")
                f.write(f"tau_{t} = {tau[t].X}\n\n")

        tau = [tau[t].X for t in range(1440)]

        # Create the plot
        plt.figure(figsize=(10, 5))

        plt.plot(range(1440), tau, color='orange', label='EWH temperature')

        # Add labels and title
        plt.xlabel('Time (minutes)')
        plt.ylabel('Power')
        plt.title('Power over Time')

        plt.legend()
        plt.show()

    if HVAC:
        sum = 0

        with open("feasible_hvac.txt", "w") as f:
            for t in range(1440):
                f.write(f"s_ac_{t} = {sAC[t].X}\n")
                f.write(f"theta_{t} = {theta[t].X}\n")

        theta = [theta[t].X for t in range(1440)]
        sAC = [sAC[t].X for t in range(1440)]

        # Create the plot
        plt.figure(figsize=(10, 5))

        plt.plot(range(1440), theta, color='blue', label='Indoor temperature')
        plt.plot(range(1440), sAC, color='orange', label='AC')

        # Add labels and title
        plt.xlabel('Time (minutes)')
        plt.ylabel('Power')
        plt.title('Power over Time')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()                


    
    if OVERALL:
        with open("feasible_overall.txt", "w") as f:
            for t in range(1440):
                f.write(f"p_G2H_{t} = {p_G2H[t].X}\n")
                f.write(f"p_H2G_{t} = {p_H2G[t].X}\n")
                f.write(f"Pv gen_{t} = {data.data_sm4['PV Energy Generation'][inTimeIntervals(t, data.data_sm4['Time Interval'])]}\n")
                f.write(f"p_B2H_{t} = {p_x2H[t][0].X}\n")
                f.write(f"p_H2B_{t} = {p_H2x[t][0].X}\n")
                f.write(f"p_V2H_{t} = {p_x2H[t][1].X if p_x2H[t][1] is not None else None}\n")
                f.write(f"p_H2V_{t} = {p_H2x[t][1].X if p_H2x[t][1] is not None else None}\n\n")

                f.write(f"E_tx_{t}_B = {E_tx[t][0].X}\n")
                f.write(f"E_tx_{t}_V = {E_tx[t][1].X if E_tx[t][1] is not None else None}\n\n")
                  
                for i in range(3):
                    f.write(f"s_shift_{t}_{i} = {s_shift[t][i].X}\n")
                    f.write(f"p_shift_{t}_{i} = {p_shift[t][i].X}\n\n")
                f.write(f"nu_{t} = {nu[t].X}\n")
                f.write(f"tau_{t} = {tau[t].X}\n\n")
                f.write(f"s_ac_{t} = {sAC[t].X}\n")
                f.write(f"theta_{t} = {theta[t].X}\n\n\n")

        # Get all variables
        variables = m.getVars()

        # Count continuous and binary variables
        num_continuous = 0
        num_binary = 0
        for v in variables:
            if v.VType == GRB.CONTINUOUS:
                num_continuous += 1

            if v.VType == GRB.BINARY:
                num_binary += 1

        # Get all constraints
        constraints = m.getConstrs()
        num_constraints = len(constraints)

        # Print the results
        print(f"Number of continuous variables: {num_continuous}")
        print(f"Number of binary variables: {num_binary}")
        print(f"Number of constraints: {num_constraints}")

        output_PH2G = [p_H2G[t].X for t in range(1440)]
        output_PG2H = [p_G2H[t].X for t in range(1440)]

        # Create the plot
        plt.figure(figsize=(10, 5))

        # Plot PG2H in blue
        plt.plot(range(1440), output_PG2H, color='blue', label='PG2H')

        # Plot PH2G in orange
        plt.plot(range(1440), output_PH2G, color='orange', label='PH2G')

        # Add labels and title
        plt.xlabel('Time (minutes)')
        plt.ylabel('Power')
        plt.title('Power over Time')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()

        output_P_DW = [p_shift[t][0].X for t in range(1440)]
        output_P_LW = [p_shift[t][1].X for t in range(1440)]
        output_P_CD = [p_shift[t][2].X for t in range(1440)]

        # Create the plot
        plt.figure(figsize=(10, 5))

        plt.plot(range(1440), output_P_DW, color='blue', label='PDW')
        plt.plot(range(1440), output_P_LW, color='orange', label='PLW')
        plt.plot(range(1440), output_P_CD, color='green', label='PCD')

        # Add labels and title
        plt.xlabel('Time (minutes)')
        plt.ylabel('Power')
        plt.title('Power over Time')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()

        output_E_B = [E_tx[t][0].X for t in range(1440)]
        output_E_V = [E_tx[t][1].X if E_tx[t][1] is not None else None for t in range(1440)]

        # Create the plot
        plt.figure(figsize=(10, 5))

        plt.plot(range(1440), output_E_B, color='blue', label='Battery')
        plt.plot(range(1440), output_E_V, color='orange', label='EV')

        # Add labels and title
        plt.xlabel('Time (minutes)')
        plt.ylabel('Power')
        plt.title('Power over Time')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()


        theta = [theta[t].X for t in range(1440)]
        tau = [tau[t].X for t in range(1440)]

        # Create the plot
        plt.figure(figsize=(10, 5))

        plt.plot(range(1440), theta, color='blue', label='Indoor temperature')
        plt.plot(range(1440), tau, color='orange', label='EWH temperature')

        # Add labels and title
        plt.xlabel('Time (minutes)')
        plt.ylabel('Power')
        plt.title('Power over Time')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()


        # p_H2B = [p_H2x[t][0].X for t in range(1440)]
        # p_B2H = [p_x2H[t][0].X for t in range(1440)]

        # p_H2V = [p_H2x[t][1].X for t in range(1440)]
        # p_V2H = [p_x2H[t][1].X for t in range(1440)]

        # # Create the plot
        # plt.figure(figsize=(10, 5))

        # plt.plot(range(1440), p_H2B, color='blue', label='House to Battery')
        # plt.plot(range(1440), p_B2H, color='orange', label='Battery to House')

        # plt.plot(range(1440), p_H2V, color='green', label='House to EV')
        # plt.plot(range(1440), p_V2H, color='pink', label='EV to House')

        # # Add labels and title
        # plt.xlabel('Time (minutes)')
        # plt.ylabel('Power')
        # plt.title('Power over Time')

        # # Add a legend
        # plt.legend()

        # # Show the plot
        # plt.show()


        theta_outdoor = [data.data_sm11["theta_t_ext"][inTimeIntervals(t, data.data_sm11["Time Interval"])] for t in range(1440)]
        sAC = [sAC[t].X for t in range(1440)]

        # Create the plot
        plt.figure(figsize=(10, 5))

        plt.plot(range(1440), theta, color='blue', label='indoor temperature')
        plt.plot(range(1440), theta_outdoor, color='orange', label='outdoor temperature')
        plt.plot(range(1440), sAC, color='green', label='sAC')

        # Add labels and title
        plt.xlabel('Time (minutes)')
        plt.ylabel('Power')
        plt.title('Power over Time')

        # Add a legend
        plt.legend()

        plt.show()

        # print(f"Total costs: {sum}")


# print(Tx)
# print(Tx[0])
# print(Tx[1])
