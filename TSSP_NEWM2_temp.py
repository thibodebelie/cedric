import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from datetime import timedelta

STEPS = ["Step_1_Unload", "Step_2_Transport", "Step_3_Infeed"]
STEP_LABELS = {"Step_1_Unload": "Unload Gate", "Step_2_Transport": "Transport Cart", "Step_3_Infeed": "Infeed Station"}
SLACKS = {"Step_1_Unload": 10, "Step_2_Transport": 7, "Step_3_Infeed": 10}
RESOURCES = {"Step_1_Unload": 4, "Step_2_Transport": 3, "Step_3_Infeed": 3}
B = 30
INPUT_FILE = "Bootstrapped_Baggage_5_Scenarios.csv"

def format_time(minutes_offset, base_datetime):
    ts = base_datetime + timedelta(minutes=float(minutes_offset))
    return ts.strftime("%#m/%#d/%Y %H:%M")

def solve_tssp(scenario_ids):
    # Load All scenario data
    df_all = pd.read_csv(INPUT_FILE)
    df_scens = df_all[df_all['Scenario_ID'].isin(scenario_ids)].copy()
    
    if df_scens.empty: return None, None, None, None, None, None
    
    df_scens['Ha Block'] = pd.to_datetime(df_scens['Ha Block'])
    base_time = df_scens['Ha Block'].min()
    
    SCENARIOS = {}
    FLIGHT_LIST = sorted(df_scens['Flight'].unique())
    num_flights = len(FLIGHT_LIST) 
    num_scenarios = len(scenario_ids)
    
    for sid in scenario_ids:
        df_sid = df_scens[df_scens['Scenario_ID'] == sid]
        SCENARIOS[sid] = {}        
        for _, row in df_sid.iterrows():
            arrival_min = int(round((row['Ha Block'] - base_time).total_seconds() / 60))
            SCENARIOS[sid][row["Flight"]] = {
                "arrival": arrival_min,
                "Step_1_Unload": int(round(row["Duration_S1"])),
                "Step_2_Transport": int(round(row["Duration_S2"])),
                "Step_3_Infeed": int(round(row["Duration_S3"]))
            }

    # Calculate earliest possible arrival for each flight across all scenarios
    # creates safe lower bound for time-indexed variables
    FLIGHT_MIN_ARRIVALS = {f: min(SCENARIOS[sid][f]["arrival"] for sid in scenario_ids) for f in FLIGHT_LIST}

    A_max = max(f_data["arrival"] for s_data in SCENARIOS.values() for f_data in s_data.values())
    D_max_sum = sum(max(f_data[s] for s_data in SCENARIOS.values() for f_data in s_data.values()) for s in STEPS)
    L_sum = sum(SLACKS.values())
    T_MAX = A_max + D_max_sum + L_sum + B 

    model = gp.Model("TSSP_Robust_Fixed")

    # ----- FIRST STAGE -----
    # "here-and-now" decisions made before uncertainty
    # x[f,s,m,t]: binary
    # ST_plan[f,s]: continuous
    x = {}
    ST_plan = {}
    AVG_DURATIONS_S3 = {}
    AVG_ARRIVALS = {}

    # Auxiliary variable for Model2 for max handling time
    Z = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Max_Planned_Handling_Time")

    for f in FLIGHT_LIST:
        # Use the global earliest arrival for this specific flight
        global_min = FLIGHT_MIN_ARRIVALS[f]
        # Average duration of step 3 across all scenarios
        AVG_DURATIONS_S3[f] = sum(SCENARIOS[sid][f]["Step_3_Infeed"] for sid in scenario_ids) / len(scenario_ids)
        # Average arrival time across all scenarios
        AVG_ARRIVALS[f] = sum(SCENARIOS[sid][f]["arrival"] for sid in scenario_ids) / len(scenario_ids)
        
        for s in STEPS:
            ST_plan[f, s] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"ST_plan_{f}_{s}")
            
            for m in range(1, RESOURCES[s] + 1):
                for t in range(global_min, T_MAX):
                    x[f, s, m, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{f}_{s}_{m}_{t}")

            # Define ST_plan and assignment constraint
            model.addConstr(ST_plan[f, s] == gp.quicksum(t * x[f, s, m, t] for m in range(1, RESOURCES[s]+1) for t in range(global_min, T_MAX)))
            model.addConstr(gp.quicksum(x[f, s, m, t] for m in range(1, RESOURCES[s]+1) for t in range(global_min, T_MAX)) == 1)
            
        # Link Z to the handling time of this specific flight
        model.addConstr(Z >= (ST_plan[f, "Step_3_Infeed"] + AVG_DURATIONS_S3[f] - AVG_ARRIVALS[f]), name=f"Max_Handling_Link_{f}")

    # ----- SECOND STAGE -----
    # "wait-and-see" decisions made after uncertainty
    # x_real[sid,f,s,m,t]: binary for realized assignment in scenario sid
    # ST_real[sid,f,s]: continuous for realized start time in scenario sid
    # delta[sid,f,s]: continuous for adjustment to planned start time in scenario sid
    x_real = {}      
    ST_real = {}    
    delta = {}      
    
    for sid in scenario_ids:
        for f in FLIGHT_LIST:
            global_min = FLIGHT_MIN_ARRIVALS[f]

            for s in STEPS:
                ST_real[sid, f, s] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"ST_real_{sid}_{f}_{s}")
                delta[sid, f, s] = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"delta_{sid}_{f}_{s}")
                
                model.addConstr(ST_real[sid, f, s] == ST_plan[f, s] + delta[sid, f, s])

                for m in range(1, RESOURCES[s] + 1):
                    for t in range(global_min, T_MAX):
                        x_real[sid, f, s, m, t] = model.addVar(vtype=GRB.BINARY, name=f"x_real_{sid}_{f}_{s}_{m}_{t}")

                model.addConstr(ST_real[sid, f, s] == gp.quicksum(t * x_real[sid, f, s, m, t] 
                                for m in range(1, RESOURCES[s]+1) for t in range(global_min, T_MAX)))
                
                # Same machine as first stage
                # Time ranges match perfectly (global_min), so summation is safe.
                for m in range(1, RESOURCES[s] + 1):
                    model.addConstr(gp.quicksum(x_real[sid, f, s, m, t] for t in range(global_min, T_MAX)) == 
                                   gp.quicksum(x[f, s, m, t] for t in range(global_min, T_MAX)))

        # Capacity constraints
        # Ensure no machine is overbooked in any scenario at any time
        for s in STEPS:
            for m in range(1, RESOURCES[s] + 1):
                for t in range(T_MAX):
                    busy_real = []
                    for f in FLIGHT_LIST:
                        global_min = FLIGHT_MIN_ARRIVALS[f]
                        D_omega = SCENARIOS[sid][f][s]
                        for ts in range(max(global_min, t - D_omega + 1), t + 1):
                            if (sid, f, s, m, ts) in x_real:
                                busy_real.append(x_real[sid, f, s, m, ts])
                    if busy_real:
                        model.addConstr(gp.quicksum(busy_real) <= 1, name=f"RealCap_{sid}_{s}_{m}_{t}")

        # Precedence and slack constraints
        # Ensure proper sequencing and slack allowances
        for f in FLIGHT_LIST:
            data = SCENARIOS[sid][f]
            model.addConstr(ST_real[sid, f, "Step_1_Unload"] >= data["arrival"])
            model.addConstr(ST_real[sid, f, "Step_1_Unload"] <= data["arrival"] + SLACKS["Step_1_Unload"])
            model.addConstr(ST_real[sid, f, "Step_2_Transport"] >= ST_real[sid, f, "Step_1_Unload"] + data["Step_1_Unload"])
            model.addConstr(ST_real[sid, f, "Step_2_Transport"] <= ST_real[sid, f, "Step_1_Unload"] + data["Step_1_Unload"] + SLACKS["Step_2_Transport"])
            model.addConstr(ST_real[sid, f, "Step_3_Infeed"] >= ST_real[sid, f, "Step_2_Transport"] + data["Step_2_Transport"])
            model.addConstr(ST_real[sid, f, "Step_3_Infeed"] <= ST_real[sid, f, "Step_2_Transport"] + data["Step_2_Transport"] + SLACKS["Step_3_Infeed"])

    # Objective Function
    # Part A: Max planned handling time (The "First-Stage" cost)
    # This is (1/|F|) * sum(ST_plan + D_avg - A_avg)
    planned_max_term = Z
    
    # Part B: Expected Recourse (The "Second-Stage" cost)
    # We define phi as the absolute value of delta using helper variables
    abs_delta = {}
    for sid in scenario_ids:
        for f in FLIGHT_LIST:
            abs_delta[sid, f] = model.addVar(lb=0, name=f"abs_delta_{sid}_{f}")
            # Constraints to force abs_delta = |delta|
            model.addConstr(abs_delta[sid, f] >= delta[sid, f, "Step_3_Infeed"])
            model.addConstr(abs_delta[sid, f] >= -delta[sid, f, "Step_3_Infeed"])

    expected_recourse_term = (1.0 / num_scenarios) * gp.quicksum(
        abs_delta[sid, f] for sid in scenario_ids for f in FLIGHT_LIST
    )
    
    # Final Objective: Minimize Max Plan + Average Recourse
    model.setObjective(planned_max_term + expected_recourse_term, GRB.MINIMIZE)

    # Solver parameters
    model.params.MIPGap = 0.02
    model.params.TimeLimit = 1800

    model.optimize()

    return model, x, ST_real, SCENARIOS, FLIGHT_LIST, base_time


if __name__ == "__main__":
    all_data = pd.read_csv(INPUT_FILE)
    scenario_list = sorted(all_data['Scenario_ID'].unique())
    
    # Solve TSSP model
    model, x, ST_real, SCENARIOS, FLIGHT_LIST, base_dt = solve_tssp(scenario_list)
    
    if model and model.SolCount > 0:
        # Extract first-stage assignments
        planned_assignments = []
        for (f, s, m, t), var in x.items():
            if var.X > 0.5:
                planned_assignments.append({
                    "Flight": f, 
                    "Step": s, 
                    "Machine": f"{STEP_LABELS[s]} {m}",
                    "Planned_Start": t
                })
        df_plan = pd.DataFrame(planned_assignments)

        # Extract second-stage realizations
        all_results = []
        for sid in scenario_list:
            for f in FLIGHT_LIST:
                d = SCENARIOS[sid][f]
                s1_s = ST_real[sid, f, "Step_1_Unload"].X
                s2_s = ST_real[sid, f, "Step_2_Transport"].X
                s3_s = ST_real[sid, f, "Step_3_Infeed"].X
                
                # Keep numeric for KPI logic
                res = {
                    "Scenario_ID": sid, 
                    "Flight": f,
                    "Arrival": d["arrival"],

                    "Start_Unloading": s1_s,
                    "End_Unloading": s1_s + d["Step_1_Unload"],
                    "Idle_Unloading": s1_s - d["arrival"],

                    "Start_Transport": s2_s, 
                    "End_Transport": s2_s + d["Step_2_Transport"],
                    "Idle_Transport": s2_s - (s1_s + d["Step_1_Unload"]),

                    "Start_Infeed": s3_s, 
                    "End_Infeed": s3_s + d["Step_3_Infeed"],
                    "Idle_Infeed": s3_s - (s2_s + d["Step_2_Transport"]),

                    "Total_Handling_Time": (s3_s + d["Step_3_Infeed"]) - d["arrival"],
                    "Total_Idle": (s1_s - d["arrival"]) + (s2_s - (s1_s + d["Step_1_Unload"])) + (s3_s - (s2_s + d["Step_2_Transport"]))
                }
                all_results.append(res)
        
        df_real_numeric = pd.DataFrame(all_results)

        # Pivot Master schedule 
        # Create summary table showing planned assignments for each flight and step
        pivot_plan = df_plan.pivot(index="Flight", columns="Step", values=["Machine", "Planned_Start"])
        pivot_plan.columns = [f"{col[1]}_{col[0]}" for col in pivot_plan.columns]
        pivot_plan = pivot_plan.reset_index()

        # Define desired order
        ordered_cols = ["Flight"]
        for s in STEPS:
            machine_col = f"{s}_Machine"
            start_col = f"{s}_Planned_Start"
            
            # Format time value before we finalize the table
            if start_col in pivot_plan.columns:
                pivot_plan[start_col] = pivot_plan[start_col].apply(lambda x: format_time(x, base_dt))
            
            # Add to ordering list
            if machine_col in pivot_plan.columns: ordered_cols.append(machine_col)
            if start_col in pivot_plan.columns: ordered_cols.append(start_col)

        # Apply new column order
        pivot_plan = pivot_plan[ordered_cols]

        # System KPIs
        makespans = df_real_numeric.groupby("Scenario_ID").apply(lambda g: g["End_Infeed"].max() - g["Arrival"].min(), include_groups=False)
        avg_makespan = makespans.mean()

        avg_handling = df_real_numeric["Total_Handling_Time"].mean()
        # Expected maximum handling time (= Avg of each scenario's MAX handling time)
        expected_max_handling = df_real_numeric.groupby("Scenario_ID")["Total_Handling_Time"].max().mean()
        # Overall maximum handling time (= absolute max across all scenarios)
        absolute_max_handling = df_real_numeric["Total_Handling_Time"].max()

        # Expected max idle & slack consumption
        scenario_maxes = df_real_numeric.groupby("Scenario_ID").agg({
            "Total_Idle": "max",
            "Idle_Unloading": "max",
            "Idle_Transport": "max",
            "Idle_Infeed": "max"
        })
        
        expected_max_idle = scenario_maxes["Total_Idle"].mean()
        expected_max_s1 = scenario_maxes["Idle_Unloading"].mean()
        expected_max_s2 = scenario_maxes["Idle_Transport"].mean()
        expected_max_s3 = scenario_maxes["Idle_Infeed"].mean()
        
        kpi_summary = [
            ["TSSP SYSTEM KPIs (Averages over Scenarios)", "VALUE"],
            ["Number of Scenarios Evaluated", len(scenario_list)],
            ["Expected Total Operation Makespan", f"{avg_makespan:.2f} min"],
            ["Expected Average Handling Time", f"{avg_handling:.2f} min"],
            ["Expected Maximum Handling Time", f"{expected_max_handling:.2f} min"],
            ["Overall Maximum Handling Time", f"{absolute_max_handling:.2f} min"],
            ["Expected Maximum Idle Time Observed", f"{expected_max_idle:.2f} min"],
            ["Expected Maximum Slack Consumption (S1/S2/S3)", f"{expected_max_s1:.2f}/{expected_max_s2:.2f}/{expected_max_s3:.2f} min"],
            ["", ""],
            ["SOLVER TECHNICALS", ""],
            ["Variables / Constraints", f"{model.NumVars} / {model.NumConstrs}"],
            ["Gurobi Solve Time", f"{model.Runtime:.4f} sec"],
            ["Optimality Gap", f"{model.MIPGap * 100:.4f} %"],
            ["", ""],
            ["PARAMETER VALUES", ""],
            ["Gate Capacity", RESOURCES["Step_1_Unload"]],
            ["Transport Capacity", RESOURCES["Step_2_Transport"]],
            ["Infeed Capacity", RESOURCES["Step_3_Infeed"]],
            ["Slack Unload", SLACKS["Step_1_Unload"]],
            ["Slack Transport", SLACKS["Step_2_Transport"]],
            ["Slack Infeed", SLACKS["Step_3_Infeed"]],
            ["Time Buffer (B)", B],
            ["", ""]
        ]

        # Add per-scenario performance breakdown
        kpi_summary.append(["SCENARIO PERFORMANCE BREAKDOWN", ""])
        kpi_summary.append(["Scenario ID", "Avg Handling | Max Handling"])
        
        # Calculate per-scenario metrics
        scenario_stats = df_real_numeric.groupby("Scenario_ID")["Total_Handling_Time"].agg(["mean", "max"])
        
        for sid, stats in scenario_stats.iterrows():
            kpi_summary.append([f"Scenario {sid}", f"{stats['mean']:.2f} min | {stats['max']:.2f} min"])
        
        kpi_summary.append(["", ""]) # Spacer before Resource Utilization
        
        summary_df = pd.DataFrame(kpi_summary)

        # Calculate expected utilization for each resource
        util_records = []
        for s in STEPS:
            for m in range(1, RESOURCES[s] + 1):
                m_name = f"{STEP_LABELS[s]} {m}"
                # Avg duration for this step across scenarios
                avg_d = sum(SCENARIOS[sid][f][s] for sid in scenario_list for f in FLIGHT_LIST) / (len(scenario_list) * len(FLIGHT_LIST))
                # Count how many flights assigned to this machine
                count = len(df_plan[df_plan["Machine"] == m_name])
                total_active = count * avg_d
                util_perc = (total_active / avg_makespan) * 100 if avg_makespan > 0 else 0
                util_records.append({"Resource": m_name, "Expected Utilization %": round(util_perc, 2)})
        util_df = pd.DataFrame(util_records)

        # Convert numeric times to formatted timestamps 
        df_real_formatted = df_real_numeric.copy()
        time_cols = ["Arrival", "Start_Unloading", "End_Unloading", "Start_Transport", "End_Transport", "Start_Infeed", "End_Infeed"]
        for col in time_cols:
            if col in df_real_formatted.columns:
                df_real_formatted[col] = df_real_formatted[col].apply(lambda x: format_time(x, base_dt))

        # Save all results to excel file with multiple sheets
        output_name = f"Thesis_TSSP_NEW_M2_Results_{len(scenario_list)}_Scenarios.xlsx"
        with pd.ExcelWriter(output_name, engine='openpyxl') as writer:
            pivot_plan.to_excel(writer, sheet_name="Master Resource Plan", index=False)
            summary_df.to_excel(writer, sheet_name="KPI Analysis", index=False, header=False)
            util_df.to_excel(writer, sheet_name="KPI Analysis", index=False, startrow=len(kpi_summary) + 2)
            df_real_formatted.to_excel(writer, sheet_name="Scenario Realizations", index=False)

        print(f"Stochastic Optimization Complete. Results saved to {output_name}")
    else:
        print("No solution found or solve failed.")