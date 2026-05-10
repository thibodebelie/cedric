import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
# Match the keys to your CSV column suffixes/names
STEPS = ["S1", "S2", "S3"]
STEP_LABELS = {"S1": "Unload Gate", "S2": "Transport Cart", "S3": "Infeed Station"}
RESOURCES = {"S1": 10, "S2": 10, "S3": 10} # Adjust based on your available hardware
LAMBDA_PENALTY = 2.0

def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    # Convert arrival times to relative minutes from the earliest arrival (t=0)
    df['Ha Block'] = pd.to_datetime(df['Ha Block'])
    min_time = df['Ha Block'].min()
    df['arrival_min'] = (df['Ha Block'] - min_time).dt.total_seconds() / 60
   
    scenarios = df['Scenario_ID'].unique().tolist()
    flights = df['Flight'].unique().tolist()
   
    # Build S_DATA[flight][scenario]
    s_data = {f: {} for f in flights}
    for _, row in df.iterrows():
        s_data[row['Flight']][row['Scenario_ID']] = {
            "arrival": row['arrival_min'],
            "S1": row['Duration_S1'],
            "S2": row['Duration_S2'],
            "S3": row['Duration_S3']
        }
       
    # Create base durations (averages) for the first-stage "Planned" duration
    base_durations = {}
    for f in flights:
        base_durations[f] = {
            s: df[df['Flight'] == f][f'Duration_{s}'].mean() for s in STEPS
        }
       
    return s_data, flights, scenarios, base_durations, min_time

# --- 2. SOLVER ---
def solve_stochastic_thesis(input_csv):
    S_DATA, FLIGHTS, SCENARIOS, BASE_DURATIONS, START_TS = load_data_from_csv(input_csv)
    PROB = 1.0 / len(SCENARIOS) # Assume equal probability if not specified
   
    T_MAX = 200 # Extend as needed
    model = gp.Model("Stochastic_Baggage_Optimization")
    model.Params.OutputFlag = 1 # Turned on to see progress

    # Variables
    x = {}
    for f in FLIGHTS:
        for s in STEPS:
            # Create machines based on the RESOURCES config for each specific step
            for m in range(1, RESOURCES[s] + 1):
                for t in range(T_MAX):
                    x[f, s, m, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{f}_{s}_{m}_{t}")

    # delta[f, s, omega]: Delay for flight f at step s in scenario omega
    delta = model.addVars(FLIGHTS, STEPS, SCENARIOS, lb=0.0, name="delta")

    # 1. Each flight/step must be assigned to exactly one machine and one start time
    for f in FLIGHTS:
        for s in STEPS:
            model.addConstr(
                gp.quicksum(x[f, s, m, t] for m in range(1, RESOURCES[s] + 1) for t in range(T_MAX)) == 1
            )

    # 2. Resource Capacity (No two flights on the same machine at the same time)
    for s in STEPS:
        for m in range(1, RESOURCES[s] + 1):
            for t in range(T_MAX):
                busy_vars = []
                for f in FLIGHTS:
                    dur = int(round(BASE_DURATIONS[f][s]))
                    for ts in range(max(0, t - dur + 1), t + 1):
                        if (f, s, m, ts) in x:
                            busy_vars.append(x[f, s, m, ts])
                if busy_vars:
                    model.addConstr(gp.quicksum(busy_vars) <= 1)

    # 3. Scenario-based Temporal Constraints (The recourse logic)
    for f in FLIGHTS:
        for o in SCENARIOS:
            # helper to get planned start time
            def p_start(fl, st):
                return gp.quicksum(t * x[fl, st, m, t] for m in range(1, RESOURCES[st] + 1) for t in range(T_MAX))

            # Actual Start = Planned Start + Delta
            act_s1 = p_start(f, "S1") + delta[f, "S1", o]
            act_s2 = p_start(f, "S2") + delta[f, "S2", o]
            act_s3 = p_start(f, "S3") + delta[f, "S3", o]

            # Sequence constraints per scenario
            model.addConstr(act_s1 >= S_DATA[f][o]["arrival"])
            model.addConstr(act_s2 >= act_s1 + S_DATA[f][o]["S1"])
            model.addConstr(act_s3 >= act_s2 + S_DATA[f][o]["S2"])

    # Objective: Minimize (Planned Completion Time) + (Penalty * Expected Delay)
    total_planned_time = gp.quicksum(p_start(f, "S3") for f in FLIGHTS)
    total_expected_delay = gp.quicksum(PROB * delta[f, s, o] for f in FLIGHTS for s in STEPS for o in SCENARIOS)
   
    model.setObjective(total_planned_time + LAMBDA_PENALTY * total_expected_delay, GRB.MINIMIZE)
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        results = []
        for f in FLIGHTS:
            for s in STEPS:
                for m in range(1, RESOURCES[s] + 1):
                    for t in range(T_MAX):
                        if x[f, s, m, t].X > 0.5:
                            avg_d = sum(delta[f, s, o].X * PROB for o in SCENARIOS)
                            results.append({
                                "Flight": f, "Step": s, "Machine": f"{STEP_LABELS[s]} {m}",
                                "Planned_Start": t, "Duration": round(BASE_DURATIONS[f][s], 1),
                                "Expected_Delay": round(avg_d, 2),
                                "Base_TS": START_TS
                            })
        return pd.DataFrame(results)
    return None

# --- 3. RUN ---
if __name__ == "__main__":
    # Ensure the CSV exists in your directory
    input_file = 'Bootstrapped_Baggage_Scenarios_baseline.csv'
    df_results = solve_stochastic_thesis(input_file)
   
    if df_results is not None:
        # Convert relative minutes back to timestamps for the Gantt
        df_results["Start_DT"] = df_results.apply(lambda r: r["Base_TS"] + timedelta(minutes=r["Planned_Start"]), axis=1)
        df_results["Finish_DT"] = df_results.apply(lambda r: r["Base_TS"] + timedelta(minutes=r["Planned_Start"] + r["Duration"]), axis=1)
       
        fig = px.timeline(df_results, x_start="Start_DT", x_end="Finish_DT", y="Machine", color="Flight",
                          hover_data=["Expected_Delay"], title="Optimized Robust Schedule from CSV Data")
        fig.show()
        df_results.to_excel("Stochastic_CSV_Results.xlsx", index=False)