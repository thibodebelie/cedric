import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import os

# --- 1. Configuration (Stayed consistent with your deterministic setup) ---
STEPS = ["Step_1_Unload", "Step_2_Transport", "Step_3_Infeed"]
STEP_LABELS = {
    "Step_1_Unload": "Unload Gate",
    "Step_2_Transport": "Transport Cart",
    "Step_3_Infeed": "Infeed Station"
}

SLACKS = {"Step_1_Unload": 10, "Step_2_Transport": 7, "Step_3_Infeed": 10}
RESOURCES = {"Step_1_Unload": 4, "Step_2_Transport": 3, "Step_3_Infeed": 3}
INPUT_FILE = "Bootstrapped_Baggage_Scenarios_baseline_hours.csv"

def solve_tssp(scenario_ids):
    # Load all relevant scenarios
    df = pd.read_csv(INPUT_FILE)
    df_all = df[df['Scenario_ID'].isin(scenario_ids)].copy()
    if df_all.empty: return None, None
    
    # Pre-processing Data for TSSP
    FLIGHTS = df_all['Flight'].unique()
    OMEGA = scenario_ids
    
    # Store scenario-specific parameters
    A_wf = {} # Arrival times
    D_wfs = {} # Durations
    
    for scen_id in OMEGA:
        df_scen = df_all[df_all['Scenario_ID'] == scen_id].copy()
        df_scen['Ha Block'] = pd.to_datetime(df_scen['Ha Block'])
        base_time = df_scen['Ha Block'].min()
        
        for _, row in df_scen.iterrows():
            f = row["Flight"]
            arrival_min = int(round((row['Ha Block'] - base_time).total_seconds() / 60))
            A_wf[scen_id, f] = arrival_min
            for s in STEPS:
                # Map Duration_S1, S2, S3 to Steps
                col_name = f"Duration_S{STEPS.index(s)+1}"
                D_wfs[scen_id, f, s] = int(round(row[col_name]))

    # Calculate T_MAX (using your exact logic, but generalized for global max)
    A_max = max(A_wf.values())
    # Max duration per step across ALL scenarios for safety
    D_max_sum = sum(max(D_wfs[w, f, s] for w in OMEGA for f in FLIGHTS) for s in STEPS)
    L_sum = sum(SLACKS.values())
    B = 30 
    T_MAX = A_max + D_max_sum + L_sum + B

    model = gp.Model("TSSP_Baggage_Handling")

    # --- 4.8.1 First-Stage Decision Variables ---
    # x_fsmt (Scenario independent assignment)
    x = {}
    for f in FLIGHTS:
        # Flight cannot start before its EARLIEST arrival across scenarios
        min_arrival = min(A_wf[w, f] for w in OMEGA)
        for s in STEPS:
            for m in range(1, RESOURCES[s] + 1):
                for t in range(min_arrival, T_MAX):
                    x[f, s, m, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{f}_{s}_{m}_{t}")

    # Planned Start Time ST_fs
    ST_plan = {}
    for f in FLIGHTS:
        for s in STEPS:
            ST_plan[f, s] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"ST_plan_{f}_{s}")
            # Definition (4.7): ST_fs = sum(m,t) t * x_fsmt
            model.addConstr(ST_plan[f, s] == gp.quicksum(t * x[f, s, m, t]
                            for m in range(1, RESOURCES[s] + 1)
                            for t in range(T_MAX) if (f, s, m, t) in x))

    # --- 4.8.2 Second-Stage Decision Variables ---
    ST_real = {} # ST^omega_fs
    delta = {}   # delta^omega_fs (Recourse)
    
    for w in OMEGA:
        for f in FLIGHTS:
            for s in STEPS:
                ST_real[w, f, s] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"ST_real_{w}_{f}_{s}")
                delta[w, f, s] = model.addVar(lb=-GRB.INFINITY, vtype=GRB.INTEGER, name=f"delta_{w}_{f}_{s}")
                
                # Equation: ST^w_fs = ST_fs + delta^w_fs
                model.addConstr(ST_real[w, f, s] == ST_plan[f, s] + delta[w, f, s])

    # --- 4.8.2 Constraints ---

    # (4.8) Assignment: Once per stage (Scenario independent)
    for f in FLIGHTS:
        for s in STEPS:
            model.addConstr(gp.quicksum(x[f, s, m, t] 
                            for m in range(1, RESOURCES[s] + 1)
                            for t in range(T_MAX) if (f, s, m, t) in x) == 1)

    # (4.9) Capacity: Based on planned assignment x, but realized duration D^w_fs
    for w in OMEGA:
        for s in STEPS:
            for m in range(1, RESOURCES[s] + 1):
                for t in range(T_MAX):
                    busy_sum = []
                    for f in FLIGHTS:
                        duration = D_wfs[w, f, s]
                        for ts in range(max(0, t - duration + 1), t + 1):
                            if (f, s, m, ts) in x:
                                busy_sum.append(x[f, s, m, ts])
                    if busy_sum:
                        model.addConstr(gp.quicksum(busy_sum) <= 1)

    # (4.11a-c) Precedence and Slacks per Scenario
    for w in OMEGA:
        for f in FLIGHTS:
            # Step 1
            model.addConstr(ST_real[w, f, "Step_1_Unload"] >= A_wf[w, f])
            model.addConstr(ST_real[w, f, "Step_1_Unload"] <= A_wf[w, f] + SLACKS["Step_1_Unload"])

            # Step 2
            model.addConstr(ST_real[w, f, "Step_2_Transport"] >= ST_real[w, f, "Step_1_Unload"] + D_wfs[w, f, "Step_1_Unload"])
            model.addConstr(ST_real[w, f, "Step_2_Transport"] <= ST_real[w, f, "Step_1_Unload"] + D_wfs[w, f, "Step_1_Unload"] + SLACKS["Step_2_Transport"])

            # Step 3
            model.addConstr(ST_real[w, f, "Step_3_Infeed"] >= ST_real[w, f, "Step_2_Transport"] + D_wfs[w, f, "Step_2_Transport"])
            model.addConstr(ST_real[w, f, "Step_3_Infeed"] <= ST_real[w, f, "Step_2_Transport"] + D_wfs[w, f, "Step_2_Transport"] + SLACKS["Step_3_Infeed"])

    # --- 4.8.3 Optimization Objective (4.15 Efficiency-Oriented) ---
    # Minimize Expected Total Handling Time
    n_omega = len(OMEGA)
    obj = (1.0 / n_omega) * gp.quicksum(ST_real[w, f, "Step_3_Infeed"] for w in OMEGA for f in FLIGHTS)
    model.setObjective(obj, GRB.MINIMIZE)

    model.setParam('TimeLimit', 600) # Complexity increases significantly with OMEGA
    model.optimize()

    if model.SolCount > 0:
        return model, x, ST_plan
    return model, None, None

if __name__ == "__main__":
    # Choose a subset of scenarios for Sample Average Approximation (SAA)
    all_data = pd.read_csv(INPUT_FILE)
    unique_scens = sorted(all_data['Scenario_ID'].unique())
    
    # Example: Solving for the first 5 scenarios together
    scenarios_to_solve = unique_scens[:5] 
    print(f"Solving TSSP for scenarios: {scenarios_to_solve}")
    
    tssp_model, x_vars, st_vars = solve_tssp(scenarios_to_solve)
    
    if x_vars:
        print("TSSP Solution found. Planned (First-Stage) Schedule:")
        # You can export st_vars to see the scenario-independent baseline
        for f in st_vars:
            print(f"Flight {f[0]} {f[1]} Planned Start: {st_vars[f].X}")