import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import plotly.express as px
from datetime import timedelta
from collections import defaultdict

# --- 1. Configuration (Converted to Seconds) ---
STEPS = ["Step_1_Unload", "Step_2_Transport", "Step_3_Infeed"]
STEP_LABELS = {
    "Step_1_Unload": "Unload Gate",
    "Step_2_Transport": "Transport Cart",
    "Step_3_Infeed": "Infeed Station"
}

# Slacks converted to integer seconds
SLACKS_SEC = {"S1_Limit": 10 * 60, "S2_Limit": 7 * 60, "S3_Limit": 10 * 60}

# Resources
RESOURCES = {"Step_1_Unload": 4, "Step_2_Transport": 3, "Step_3_Infeed": 3}

INPUT_FILE = "Bootstrapped_Baggage_Scenarios_baseline_hours2.csv"

def get_flight_data_seconds(scenario_id=1):
    df = pd.read_csv(INPUT_FILE)
    df_scen = df[df['Scenario_ID'] == scenario_id].copy()
    df_scen['Ha Block'] = pd.to_datetime(df_scen['Ha Block'])
    base_time = df_scen['Ha Block'].min()
   
    flights_dict = {}
    for _, row in df_scen.iterrows():
        # Convert all floats to integer seconds
        arrival_sec = int(round((row['Ha Block'] - base_time).total_seconds()))
       
        flights_dict[row["Flight"]] = {
            "arrival": arrival_sec,
            "Step_1_Unload": int(round(row["Duration_S1"] * 60)),
            "Step_2_Transport": int(round(row["Duration_S2"] * 60)),
            "Step_3_Infeed": int(round(row["Duration_S3"] * 60)),
            "original_dt": row['Ha Block']
        }
    return flights_dict, base_time

def solve_thesis_deterministic_seconds():
    FLIGHTS, BASE_TIME = get_flight_data_seconds(scenario_id=1)
    model = gp.Model("Thesis_Discrete_Seconds")
   
    # 1. Variables
    start_time = {} # Integer: Start time in seconds
    assign = {}     # Binary: 1 if flight f at step s uses machine m
    x = {}          # Binary: 1 if flight f starts step s on machine m at second ts

    # Track usage for the capacity constraint efficiently
    # key: (step, machine, second_t), value: [list of binary vars x]
    usage_map = defaultdict(list)

    print("Creating variables and mapping time-slots...")
    for f in FLIGHTS:
        # Windowing: Only create variables where the flight can actually exist
        # This keeps the integer-second model size manageable.
        earliest_start = FLIGHTS[f]["arrival"]
        latest_start = earliest_start + sum(SLACKS_SEC.values()) + 1800 # 30 min buffer

        for s in STEPS:
            dur = FLIGHTS[f][s]
            start_time[f, s] = model.addVar(lb=0, vtype=GRB.INTEGER, name=f"st_{f}_{s}")
           
            for m in range(1, RESOURCES[s] + 1):
                assign[f, s, m] = model.addVar(vtype=GRB.BINARY, name=f"assign_{f}_{s}_{m}")
               
                for ts in range(earliest_start, latest_start):
                    var = model.addVar(vtype=GRB.BINARY, name=f"x_{f}_{s}_{m}_{ts}")
                    x[f, s, m, ts] = var
                   
                    # Map this variable to all seconds it would occupy the machine
                    # This replaces the need for the slow triple-nested 'busy' loop
                    for second_occupied in range(ts, ts + dur):
                        usage_map[s, m, second_occupied].append(var)

    # 2. Constraint: Assignment
    # Links x[f,s,m,ts] to both start_time and assign variables
    for f in FLIGHTS:
        for s in STEPS:
            relevant_vars = [v for (fl, st, m, ts), v in x.items() if fl == f and st == s]
           
            # Each flight/step must be assigned to exactly one machine/start-second
            model.addConstr(gp.quicksum(relevant_vars) == 1)
           
            # Link start_time to the specific second chosen in x
            model.addConstr(start_time[f, s] == gp.quicksum(ts * v for (fl, st, m, ts), v in x.items()
                                                            if fl == f and st == s))
           
            # Link assign[m] to x
            for m in range(1, RESOURCES[s] + 1):
                model.addConstr(assign[f, s, m] == gp.quicksum(v for (fl, st, mach, ts), v in x.items()
                                                               if fl == f and st == s and mach == m))

    # 3. Constraint: Capacity (The "Original" Summation Form)
    # Using the usage_map to build: sum(x) <= 1 for every machine at every second
    print("Adding capacity constraints...")
    for (s, m, t), busy_list in usage_map.items():
        if len(busy_list) > 1:
            model.addConstr(gp.quicksum(busy_list) <= 1)

    # 4. Constraint: Precedence & Distributed Slacks (Using Seconds)
    for f in FLIGHTS:
        # Step 1: Unload
        model.addConstr(start_time[f, "Step_1_Unload"] >= FLIGHTS[f]["arrival"])
        model.addConstr(start_time[f, "Step_1_Unload"] <= FLIGHTS[f]["arrival"] + SLACKS_SEC["S1_Limit"])

        # Step 2: Transport
        s1_end = start_time[f, "Step_1_Unload"] + FLIGHTS[f]["Step_1_Unload"]
        model.addConstr(start_time[f, "Step_2_Transport"] >= s1_end)
        model.addConstr(start_time[f, "Step_2_Transport"] <= s1_end + SLACKS_SEC["S2_Limit"])

        # Step 3: Infeed
        s2_end = start_time[f, "Step_2_Transport"] + FLIGHTS[f]["Step_2_Transport"]
        model.addConstr(start_time[f, "Step_3_Infeed"] >= s2_end)
        model.addConstr(start_time[f, "Step_3_Infeed"] <= s2_end + SLACKS_SEC["S3_Limit"])

    # 5. Objective: Minimize total handling time (Sum of end-of-step-3 times in seconds)
    obj = gp.quicksum(start_time[f, "Step_3_Infeed"] for f in FLIGHTS)
    model.setObjective(obj, GRB.MINIMIZE)

    model.setParam('TimeLimit', 300)
    model.optimize()

    # if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
    if model.SolCount > 0:
        res = []
        for f in FLIGHTS:
            for s in STEPS:
                m_idx = next(m for m in range(1, RESOURCES[s] + 1) if assign[f, s, m].X > 0.5)
                st = start_time[f, s].X
                res.append({
                    "Flight": f, "Step": s,
                    "Machine": f"{STEP_LABELS[s]} {m_idx}",
                    "Start_Sec": st, "Duration_Sec": FLIGHTS[f][s],
                    "Finish_Sec": st + FLIGHTS[f][s], "Arrival_Sec": FLIGHTS[f]["arrival"]
                })
        return pd.DataFrame(res), BASE_TIME
    return None, None

def plot_gantt_seconds(df, base_dt):
    plot_df = df.copy()
    plot_df["Start_DT"] = plot_df["Start_Sec"].apply(lambda x: base_dt + timedelta(seconds=x))
    plot_df["Finish_DT"] = plot_df["Finish_Sec"].apply(lambda x: base_dt + timedelta(seconds=x))
   
    y_order = [f"{STEP_LABELS[s]} {m}" for s in STEPS for m in range(1, RESOURCES[s] + 1)]
   
    fig = px.timeline(plot_df, x_start="Start_DT", x_end="Finish_DT", y="Machine", color="Flight",
                      category_orders={"Machine": y_order},
                      title="Deterministic Optimization (1-Second Integer Precision)")

    fig.update_yaxes(autorange="reversed")
    fig.show()

# --- MAIN EXECUTION BLOCK WITH EXCEL GENERATION ---
if __name__ == "__main__":
    df_res, base_dt, model = solve_thesis_deterministic_seconds()
   
    if df_res is not None:
        # 1. Convert everything to minutes for the KPI logic
        df_min = df_res.copy()
        for col in ["Start_Sec", "Finish_Sec", "Arrival_Sec", "Duration_Sec"]:
            df_min[col.replace("_Sec", "")] = df_min[col] / 60.0

        # 2. Data Preparation (Pivoting)
        pivot_df = df_min.pivot(index=["Flight", "Arrival"], columns="Step", values=["Machine", "Start", "Finish"])
        pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
        pivot_df = pivot_df.reset_index()
       
        # 3. Calculate KPIs
        pivot_df["Total_Handling_Time"] = pivot_df["Step_3_Infeed_Finish"] - pivot_df["Arrival"]
        pivot_df["Idle time Unload"] = pivot_df["Step_1_Unload_Start"] - pivot_df["Arrival"]
        pivot_df["Idle time Transport"] = pivot_df["Step_2_Transport_Start"] - pivot_df["Step_1_Unload_Finish"]
        pivot_df["Idle time Infeed"] = pivot_df["Step_3_Infeed_Start"] - pivot_df["Step_2_Transport_Finish"]
        pivot_df["Total Idle Time"] = pivot_df["Idle time Unload"] + pivot_df["Idle time Transport"] + pivot_df["Idle time Infeed"]

        max_handling_val = pivot_df["Total_Handling_Time"].max()
        max_idle_val = pivot_df["Total Idle Time"].max()
        max_idle_flight = pivot_df.loc[pivot_df["Total Idle Time"].idxmax(), "Flight"]
       
        # Utilization
        total_span_min = df_min["Finish"].max() - df_min["Arrival"].min()
        util_records = []
        for s in STEPS:
            for m in range(1, RESOURCES[s] + 1):
                m_name = f"{STEP_LABELS[s]} {m}"
                active_min = df_min[df_min["Machine"] == m_name]["Duration"].sum()
                util_percent = (active_min / total_span_min) * 100 if total_span_min > 0 else 0
                util_records.append({"Resource": m_name, "Utilization %": round(util_percent, 2)})
        util_df = pd.DataFrame(util_records)

        # 4. Summary Table
        kpi_summary = [
            ["OVERALL SYSTEM KPIs (Minutes)", "VALUE"],
            ["Total Operation Makespan", f"{total_span_min:.2f} min"],
            ["Average Handling Time", f"{pivot_df['Total_Handling_Time'].mean():.2f} min"],
            ["Maximum Handling Time", f"{max_handling_val:.2f} min"],
            ["Average Total Idle Time", f"{pivot_df['Total Idle Time'].mean():.2f} min"],
            ["Max Idle Time Observed", f"{max_idle_val:.2f} min / {max_idle_flight}"],
            ["Max Slack Consumption (S1/S2/S3)", f"{pivot_df['Idle time Unload'].max():.2f}/{pivot_df['Idle time Transport'].max():.2f}/{pivot_df['Idle time Infeed'].max():.2f} min"],
            ["", ""],
            ["SOLVER TECHNICALS", ""],
            ["Variables / Constraints", f"{model.NumVars} / {model.NumConstrs}"],
            ["Gurobi Solve Time", f"{model.Runtime:.4f} sec"]
        ]
        summary_df = pd.DataFrame(kpi_summary)

        # 5. Excel Export
        final_cols = {
            "Flight": "Flight Name", "Arrival": "Arrival Time",
            "Step_1_Unload_Machine": "Gate", "Step_1_Unload_Start": "Start Time Unload", "Step_1_Unload_Finish": "End Time Unload", "Idle time Unload": "Idle time Unload",
            "Step_2_Transport_Machine": "Cart", "Step_2_Transport_Start": "Start Time Transport", "Step_2_Transport_Finish": "End Time Transport", "Idle time Transport": "Idle time Transport",
            "Step_3_Infeed_Machine": "Infeed Station", "Step_3_Infeed_Start": "Start Time Infeed", "Step_3_Infeed_Finish": "End Time Infeed", "Idle time Infeed": "Idle time Infeed",
            "Total Idle Time": "Total Idle Time", "Total_Handling_Time": "Total Handling Time"
        }
        export_df = pivot_df[list(final_cols.keys())].rename(columns=final_cols)

        file_name = "Thesis_Deterministic_Results.xlsx"
        with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
            export_df.to_excel(writer, sheet_name="Flight Schedule", index=False)
            summary_df.to_excel(writer, sheet_name="KPI Analysis", index=False, header=False)
            util_df.to_excel(writer, sheet_name="KPI Analysis", index=False, startrow=len(kpi_summary) + 2)

        print(f"Analysis complete. Excel generated: {file_name}")
        plot_gantt_seconds(df_res, base_dt)
    else:
        print("No solution found.")