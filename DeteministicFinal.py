import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from datetime import timedelta
import os

# --- 1. Configuration (All times in minutes for clarity, matching Methodology) ---
STEPS = ["Step_1_Unload", "Step_2_Transport", "Step_3_Infeed"]
STEP_LABELS = {
    "Step_1_Unload": "Unload Gate",
    "Step_2_Transport": "Transport Cart",
    "Step_3_Infeed": "Infeed Station"
}

# Slacks L_s from Methodology
SLACKS = {"Step_1_Unload": 10, "Step_2_Transport": 7, "Step_3_Infeed": 10}
RESOURCES = {"Step_1_Unload": 4, "Step_2_Transport": 3, "Step_3_Infeed": 3}
INPUT_FILE = "Bootstrapped_Baggage_Scenarios_baseline_hours.csv"

def solve_scenario(scenario_id):
    # Load Scenario Data
    df = pd.read_csv(INPUT_FILE)
    df_scen = df[df['Scenario_ID'] == scenario_id].copy()
    if df_scen.empty: return None, None, None
   
    df_scen['Ha Block'] = pd.to_datetime(df_scen['Ha Block'])
    base_time = df_scen['Ha Block'].min()
   
    # Parameters F, A_f, D_fs
    FLIGHTS = {}
    for _, row in df_scen.iterrows():
        arrival_min = int(round((row['Ha Block'] - base_time).total_seconds() / 60))
        FLIGHTS[row["Flight"]] = {
            "arrival": arrival_min,
            "Step_1_Unload": int(round(row["Duration_S1"])),
            "Step_2_Transport": int(round(row["Duration_S2"])),
            "Step_3_Infeed": int(round(row["Duration_S3"]))
        }

    # Calculate T_max as per Methodology 4.5.1
    A_max = max(f["arrival"] for f in FLIGHTS.values())
    D_max_sum = sum(max(f[s] for f in FLIGHTS.values()) for s in STEPS)
    L_sum = sum(SLACKS.values())
    B = 30 # Buffer parameter
    T_MAX = A_max + D_max_sum + L_sum + B

    model = gp.Model(f"Deterministic_Sce_{scenario_id}")
   
    # --- 4.5.2 Decision Variables ---
    # x_fsmt (4.6)
    x = {}
    for f in FLIGHTS:
        for s in STEPS:
            # Operational Windowing: flight cannot start before arrival or after T_MAX
            for m in range(1, RESOURCES[s] + 1):
                for t in range(FLIGHTS[f]["arrival"], T_MAX):
                    x[f, s, m, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{f}_{s}_{m}_{t}")

    # Auxiliary variables ST_fs (4.7/4.8)
    ST = {}
    for f in FLIGHTS:
        for s in STEPS:
            ST[f, s] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"ST_{f}_{s}")
            # Definition: ST_fs = sum(m)sum(t) t * x_fsmt
            model.addConstr(ST[f, s] == gp.quicksum(t * x[f, s, m, t]
                            for m in range(1, RESOURCES[s] + 1)
                            for t in range(FLIGHTS[f]["arrival"], T_MAX)
                            if (f, s, m, t) in x))

    # --- 4.5.4 Constraints ---

    # (4.3) Assignment: Each flight processed exactly once per step
    for f in FLIGHTS:
        for s in STEPS:
            model.addConstr(gp.quicksum(x[f, s, m, t]
                            for m in range(1, RESOURCES[s] + 1)
                            for t in range(FLIGHTS[f]["arrival"], T_MAX)
                            if (f, s, m, t) in x) == 1)

    # (4.4) Capacity: Prevent multiple flights on same resource at same time
    print(f"Adding capacity constraints for Scenario {scenario_id}...")
    for s in STEPS:
        for m in range(1, RESOURCES[s] + 1):
            for t in range(T_MAX):
                busy_sum = []
                for f in FLIGHTS:
                    D_fs = FLIGHTS[f][s]
                    # Inner sum: ts = max(0, t - D_fs + 1) to t
                    for ts in range(max(0, t - D_fs + 1), t + 1):
                        if (f, s, m, ts) in x:
                            busy_sum.append(x[f, s, m, ts])
                if busy_sum:
                    model.addConstr(gp.quicksum(busy_sum) <= 1)

    # (4.5a) Precedence Step 1
    for f in FLIGHTS:
        model.addConstr(ST[f, "Step_1_Unload"] >= FLIGHTS[f]["arrival"])
        model.addConstr(ST[f, "Step_1_Unload"] <= FLIGHTS[f]["arrival"] + SLACKS["Step_1_Unload"])

        # (4.5b) Precedence Step 2
        model.addConstr(ST[f, "Step_2_Transport"] >= ST[f, "Step_1_Unload"] + FLIGHTS[f]["Step_1_Unload"])
        model.addConstr(ST[f, "Step_2_Transport"] <= ST[f, "Step_1_Unload"] + FLIGHTS[f]["Step_1_Unload"] + SLACKS["Step_2_Transport"])

        # (4.5c) Precedence Step 3
        model.addConstr(ST[f, "Step_3_Infeed"] >= ST[f, "Step_2_Transport"] + FLIGHTS[f]["Step_2_Transport"])
        model.addConstr(ST[f, "Step_3_Infeed"] <= ST[f, "Step_2_Transport"] + FLIGHTS[f]["Step_2_Transport"] + SLACKS["Step_3_Infeed"])

    # --- 4.5.3.1 Optimization Objective (4.1) ---
    obj = gp.quicksum(ST[f, "Step_3_Infeed"] for f in FLIGHTS)
    model.setObjective(obj, GRB.MINIMIZE)

    model.setParam('TimeLimit', 300)
    model.optimize()

    if model.SolCount > 0:
        results = []
        for (f, s, m, t), var in x.items():
            if var.X > 0.5:
                results.append({
                    "Flight": f, "Step": s, "Machine": f"{STEP_LABELS[s]} {m}",
                    "Start": t, "Duration": FLIGHTS[f][s], "Finish": t + FLIGHTS[f][s],
                    "Arrival": FLIGHTS[f]["arrival"]
                })
        return pd.DataFrame(results), base_time, model
    return None, None, model

if __name__ == "__main__":
    # Get all unique scenarios
    all_data = pd.read_csv(INPUT_FILE)
    scenario_list = sorted(all_data['Scenario_ID'].unique())
    print(f"Found {len(scenario_list)} scenarios: {scenario_list}")

    for scen_id in scenario_list:
        df_res, base_dt, model = solve_scenario(scen_id)
       
        if df_res is not None:
            # 2. Pivoting & KPIs
            pivot_df = df_res.pivot(index=["Flight", "Arrival"], columns="Step", values=["Machine", "Start", "Finish"])
            pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
            pivot_df = pivot_df.reset_index()
           
            pivot_df["Total_Handling_Time"] = pivot_df["Step_3_Infeed_Finish"] - pivot_df["Arrival"]
            pivot_df["Idle time Unload"] = pivot_df["Step_1_Unload_Start"] - pivot_df["Arrival"]
            pivot_df["Idle time Transport"] = pivot_df["Step_2_Transport_Start"] - pivot_df["Step_1_Unload_Finish"]
            pivot_df["Idle time Infeed"] = pivot_df["Step_3_Infeed_Start"] - pivot_df["Step_2_Transport_Finish"]
            pivot_df["Total Idle Time"] = pivot_df["Idle time Unload"] + pivot_df["Idle time Transport"] + pivot_df["Idle time Infeed"]

            # Utilization Calculation
            total_span_min = df_min["Finish"].max() - df_min["Arrival"].min()
            util_records = []
            for s in STEPS:
                for m in range(1, RESOURCES[s] + 1):
                    m_name = f"{STEP_LABELS[s]} {m}"
                    active_min = df_min[df_min["Machine"] == m_name]["Duration"].sum()
                    util_percent = (active_min / total_span_min) * 100 if total_span_min > 0 else 0
                    util_records.append({"Resource": m_name, "Utilization %": round(util_percent, 2)})
            util_df = pd.DataFrame(util_records)

            # 3. Summary Table
            kpi_summary = [
                ["OVERALL SYSTEM KPIs (Minutes)", "VALUE"],
                ["Total Operation Makespan", f"{total_span_min:.2f} min"],
                ["Average Handling Time", f"{pivot_df['Total_Handling_Time'].mean():.2f} min"],
                ["Maximum Handling Time", f"{pivot_df['Total_Handling_Time'].max():.2f} min"],
                ["Average Total Idle Time", f"{pivot_df['Total Idle Time'].mean():.2f} min"],
                ["Max Idle Time Observed", f"{pivot_df['Total Idle Time'].max():.2f} min"],
                ["Max Slack Consumption (S1/S2/S3)", f"{pivot_df['Idle time Unload'].max():.2f}/{pivot_df['Idle time Transport'].max():.2f}/{pivot_df['Idle time Infeed'].max():.2f} min"],
                ["", ""],
                ["SOLVER TECHNICALS", ""],
                ["Variables / Constraints", f"{model.NumVars} / {model.NumConstrs}"],
                ["Gurobi Solve Time", f"{model.Runtime:.4f} sec"],
                ["Optimality Gap", f"{model.MIPGap:.4f} %"]
            ]
            summary_df = pd.DataFrame(kpi_summary)

            # 4. Excel Export
            final_cols = {
                "Flight": "Flight Name", "Arrival": "Arrival Time",
                "Step_1_Unload_Machine": "Gate", "Step_1_Unload_Start": "Start Time Unload", "Step_1_Unload_Finish": "End Time Unload", "Idle time Unload": "Idle time Unload",
                "Step_2_Transport_Machine": "Cart", "Step_2_Transport_Start": "Start Time Transport", "Step_2_Transport_Finish": "End Time Transport", "Idle time Transport": "Idle time Transport",
                "Step_3_Infeed_Machine": "Infeed Station", "Step_3_Infeed_Start": "Start Time Infeed", "Step_3_Infeed_Finish": "End Time Infeed", "Idle time Infeed": "Idle time Infeed",
                "Total Idle Time": "Total Idle Time", "Total_Handling_Time": "Total Handling Time"
            }
            export_df = pivot_df[list(final_cols.keys())].rename(columns=final_cols)

            file_name = f"Thesis_Deterministic_Results_Scenario_{scen_id}.xlsx"
            with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
                export_df.to_excel(writer, sheet_name="Flight Schedule", index=False)
                summary_df.to_excel(writer, sheet_name="KPI Analysis", index=False, header=False)
                util_df.to_excel(writer, sheet_name="KPI Analysis", index=False, startrow=len(kpi_summary) + 2)

            print(f"Scenario {scen_id} complete. Excel generated: {file_name}")
           
        else:
            print(f"Scenario {scen_id}: No solution found within time limit.")
       
        # Free memory
        if model:
            model.dispose()