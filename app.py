import streamlit as st
import pandas as pd
import pulp
import io
import plotly.graph_objects as go

st.set_page_config(page_title="UK Strategic Network: C-Suite Architect", layout="wide")

# --- 1. DATA FACTORY ---
@st.cache_data
def generate_rich_template():
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        pd.DataFrame({"Parameter": ["WACC", "Variable_Cost_Inflation", "Tax Rate"], "Value": [0.10, 0.03, 0.25]}).to_excel(writer, sheet_name="Constants", index=False)
        pd.DataFrame({"Supplier": ["Shenzhen_China"], "RM_Cost": [750], "Tariff_Rate": [0.20]}).to_excel(writer, sheet_name="Suppliers", index=False)
        
        pd.DataFrame({
            "Site": ["Manchester", "Birmingham", "Glasgow"],
            "Cap_Std": [40000, 55000, 30000],
            "Cap_Mega": [120000, 160000, 100000],
            "Fixed_Cost_Annual": [850000, 1350000, 650000]
        }).to_excel(writer, sheet_name="Facilities", index=False)
        
        pd.DataFrame({
            "DC_Location": ["North_Hub", "Midlands_Hub", "South_Hub"],
            "Fixed_Cost": [450000, 650000, 850000],
            "Variable_Handling_Cost": [48, 52, 68],
            "Owned_Fixed_Cost": [180000, 280000, 380000],
            "Owned_Var_Handling": [22, 24, 30],
            "Owned_CAPEX": [4500000, 6500000, 8500000]
        }).to_excel(writer, sheet_name="3PL_Nodes", index=False)
        
        years, regs = [1,2,3,4,5], ["London", "Scotland", "North_East", "South_West", "NI"]
        demand_matrix = [{"Year": y, "London": 18000*y, "Scotland": 7500*y, "North_East": 9500*y, "South_West": 8000*y, "NI": 3800*y} for y in years]
        pd.DataFrame(demand_matrix).to_excel(writer, sheet_name="Demand", index=False)
        
        pd.DataFrame([{"From": "Shenzhen_China", "Manchester": 190, "Birmingham": 160, "Glasgow": 225}]).to_excel(writer, sheet_name="Freight_Inbound", index=False)
        
        pd.DataFrame([
            {"From": "Manchester", "North_Hub": 12, "Midlands_Hub": 48, "South_Hub": 95},
            {"From": "Birmingham", "North_Hub": 48, "Midlands_Hub": 15, "South_Hub": 45},
            {"From": "Glasgow", "North_Hub": 68, "Midlands_Hub": 110, "South_Hub": 185}
        ]).to_excel(writer, sheet_name="Freight_Outbound", index=False)
        
        pd.DataFrame([
            {"From": "North_Hub", "London": 130, "Scotland": 72, "North_East": 25, "South_West": 148, "NI": 260},
            {"From": "Midlands_Hub", "London": 70, "Scotland": 148, "North_East": 62, "South_West": 82, "NI": 225},
            {"From": "South_Hub", "London": 25, "Scotland": 280, "North_East": 138, "South_West": 52, "NI": 380}
        ]).to_excel(writer, sheet_name="Last_Mile", index=False)
    return output.getvalue()

# --- 2. HEADER ---
st.title("🇬🇧 UK Strategic Network: C-Suite Architect")
st.markdown("**Objective:** A path-based MILP solver that optimizes multi-echelon network flows, CAPEX timing, and market rationalization.")

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("📥 Data Management")
    st.download_button("📥 Download Network Data", data=generate_rich_template(), file_name="Network_Specs.xlsx")
    uploaded_file = st.file_uploader("Upload Specs (.xlsx)", type=["xlsx"])
    
    st.header("🏢 Corporate Strategy")
    strategy = st.radio("Asset Policy", ["Optimize Mix", "3PL Only", "Owned Only"])
    service_mode = st.radio("Commercial Policy", ["Capture All Demand", "Profit Max (Rationalize)"])
    
    st.header("🔄 Reverse Logistics")
    sim_returns = st.slider("Global Returns (%)", 0, 30, 8) / 100.0
    sim_refurb = st.slider("Refurb Cost (£/unit)", 50, 500, 150)
    
    run_button = st.button("🚀 Solve Strategic Network", type="primary", use_container_width=True)

# --- 4. PATH-BASED SOLVER ENGINE ---
if run_button:
    source_data = uploaded_file if uploaded_file else io.BytesIO(generate_rich_template())
    try:
        df_fac = pd.read_excel(source_data, sheet_name='Facilities').set_index('Site')
        df_dem = pd.read_excel(source_data, sheet_name='Demand').set_index('Year')
        df_sup = pd.read_excel(source_data, sheet_name='Suppliers').set_index('Supplier')
        df_3pl = pd.read_excel(source_data, sheet_name='3PL_Nodes').set_index('DC_Location')
        df_f_in = pd.read_excel(source_data, sheet_name='Freight_Inbound').set_index('From')
        df_f_out = pd.read_excel(source_data, sheet_name='Freight_Outbound').set_index('From')
        df_last = pd.read_excel(source_data, sheet_name='Last_Mile').set_index('From')
        df_const = pd.read_excel(source_data, sheet_name='Constants').set_index('Parameter')
    except Exception as e:
        st.error(f"Data Schema Error: {e}"); st.stop()

    WACC, PRICE = df_const.loc['WACC', 'Value'], 2500
    years, sups, facs, dcs, regs = [1,2,3,4,5], df_sup.index.tolist(), df_fac.index.tolist(), df_3pl.index.tolist(), df_dem.columns.tolist()

    model = pulp.LpProblem("UK_C_Suite_Network", pulp.LpMaximize)

    # INFRASTRUCTURE BINARIES
    build_fac = pulp.LpVariable.dicts("Fac_Build", ((f, y, s) for f in facs for y in years for s in ['Std', 'Mega']), cat='Binary')
    build_own = pulp.LpVariable.dicts("DC_Build", ((d, y) for d in dcs for y in years), cat='Binary')
    open_3pl = pulp.LpVariable.dicts("DC_3PL", ((d, y) for d in dcs for y in years), cat='Binary')

    # PATH-BASED FLOW VARIABLES (Ensures Perfect Traceability)
    # Split into 3PL vs Owned to perfectly track distinct handling costs
    p_3pl = pulp.LpVariable.dicts("Path_3PL", ((s, f, d, r, y) for s in sups for f in facs for d in dcs for r in regs for y in years), lowBound=0)
    p_own = pulp.LpVariable.dicts("Path_Own", ((s, f, d, r, y) for s in sups for f in facs for d in dcs for r in regs for y in years), lowBound=0)

    # CONSTRAINTS
    # 1. Build Once Constraints
    for f in facs: model += pulp.lpSum([build_fac[(f, y, sz)] for y in years for sz in ['Std', 'Mega']]) <= 1
    for d in dcs: model += pulp.lpSum([build_own[(d, y)] for y in years]) <= 1
    
    for y in years:
        for r in regs:
            # Demand satisfaction
            vol = pulp.lpSum([p_3pl[(s, f, d, r, y)] + p_own[(s, f, d, r, y)] for s in sups for f in facs for d in dcs])
            if service_mode == "Capture All Demand": model += vol == df_dem.loc[y, r]
            else: model += vol <= df_dem.loc[y, r]
            
        for f in facs:
            # Factory Capacity
            cap = pulp.lpSum([build_fac[(f, yb, 'Std')] * df_fac.loc[f, 'Cap_Std'] + build_fac[(f, yb, 'Mega')] * df_fac.loc[f, 'Cap_Mega'] for yb in years if yb <= y])
            model += pulp.lpSum([p_3pl[(s, f, d, r, y)] + p_own[(s, f, d, r, y)] for s in sups for d in dcs for r in regs]) <= cap
            
        for d in dcs:
            # Strategy & DC Capacity Constraints
            if strategy == "3PL Only": model += build_own[(d, y)] == 0
            elif strategy == "Owned Only": model += open_3pl[(d, y)] == 0
            
            model += pulp.lpSum([p_3pl[(s, f, d, r, y)] for s in sups for f in facs for r in regs]) <= open_3pl[(d, y)] * 1e7
            model += pulp.lpSum([p_own[(s, f, d, r, y)] for s in sups for f in facs for r in regs]) <= pulp.lpSum([build_own[(d, yb)] for yb in years if yb <= y]) * 1e7

    # NPV CALCULATIONS (Path-by-Path Exact Costing)
    cfs = []
    for y in years:
        inf = (1 + df_const.loc['Variable_Cost_Inflation', 'Value'])**(y-1)
        net_cash = 0
        
        for s in sups:
            for f in facs:
                for d in dcs:
                    for r in regs:
                        v_3 = p_3pl[(s, f, d, r, y)]
                        v_o = p_own[(s, f, d, r, y)]
                        
                        base_cost = (df_sup.loc[s, 'RM_Cost']*1.2 + df_f_in.loc[s, f] + df_f_out.loc[f, d] + df_last.loc[d, r])
                        rev_leak = sim_returns * (df_last.loc[d, r] + sim_refurb)
                        
                        # Apply specific handling costs based on path chosen
                        cost_3pl = (base_cost + df_3pl.loc[d, 'Variable_Handling_Cost'] + rev_leak) * inf
                        cost_own = (base_cost + df_3pl.loc[d, 'Owned_Var_Handling'] + rev_leak) * inf
                        
                        net_cash += (v_3 * (PRICE - cost_3pl)) + (v_o * (PRICE - cost_own))

        fixed = pulp.lpSum([open_3pl[(d, y)] * df_3pl.loc[d, 'Fixed_Cost'] + build_own[(d, y)] * df_3pl.loc[d, 'Owned_Fixed_Cost'] for d in dcs]) + \
                pulp.lpSum([build_fac[(f, y, sz)] * df_fac.loc[f, 'Fixed_Cost_Annual'] for f in facs for sz in ['Std', 'Mega'] for yb in years if yb <= y])
        capex = pulp.lpSum([build_own[(d, y)] * df_3pl.loc[d, 'Owned_CAPEX'] for d in dcs]) + pulp.lpSum([build_fac[(f, y, sz)] * (5e6 if sz=='Std' else 12e6) for f in facs for sz in ['Std', 'Mega']])
        
        cfs.append((net_cash - fixed - capex) / ((1 + WACC)**y))

    model += pulp.lpSum(cfs)
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # --- 5. C-SUITE DASHBOARDS ---
    st.metric("Total Strategic NPV (5-Year)", f"£{pulp.value(model.objective)/1e6:.2f}M")
    t1, t2, t3, t4 = st.tabs(["🏗️ CAPEX & Build Schedule", "🚚 Volume Flow (Sankey)", "📍 Asset Utilization", "📥 Forensic CFO Ledger"])

    with t1:
        st.subheader("Infrastructure CAPEX Timeline")
        build_sched = []
        for y in years:
            for f in facs:
                for sz in ['Std', 'Mega']:
                    if pulp.value(build_fac[(f, y, sz)]) > 0.5:
                        build_sched.append({"Year": y, "Asset": f, "Action": f"Build Factory ({sz})", "CAPEX": f"£{5 if sz=='Std' else 12}M"})
            for d in dcs:
                if pulp.value(build_own[(d, y)]) > 0.5:
                    build_sched.append({"Year": y, "Asset": d, "Action": "Build Owned DC", "CAPEX": f"£{df_3pl.loc[d, 'Owned_CAPEX']/1e6:.1f}M"})
                if pulp.value(open_3pl[(d, y)]) > 0.5:
                    build_sched.append({"Year": y, "Asset": d, "Action": "Open 3PL Contract", "CAPEX": "£0M"})
        if build_sched:
            st.dataframe(pd.DataFrame(build_sched), use_container_width=True)
        else:
            st.info("No Infrastructure Actions Taken.")

    with t2:
        st.subheader("Physical Unit Routing (5-Year Volume)")
        nodes = sups + facs + dcs + regs
        n_map = {n: i for i, n in enumerate(nodes)}
        s_idx, t_idx, v_val = [], [], []
        
        for s in sups:
            for f in facs:
                v = sum([pulp.value(p_3pl[(s,f,d,r,y)]) + pulp.value(p_own[(s,f,d,r,y)]) for d in dcs for r in regs for y in years])
                if v > 1: s_idx.append(n_map[s]); t_idx.append(n_map[f]); v_val.append(v)
        for f in facs:
            for d in dcs:
                v = sum([pulp.value(p_3pl[(s,f,d,r,y)]) + pulp.value(p_own[(s,f,d,r,y)]) for s in sups for r in regs for y in years])
                if v > 1: s_idx.append(n_map[f]); t_idx.append(n_map[d]); v_val.append(v)
        for d in dcs:
            for r in regs:
                v = sum([pulp.value(p_3pl[(s,f,d,r,y)]) + pulp.value(p_own[(s,f,d,r,y)]) for s in sups for f in facs for y in years])
                if v > 1: s_idx.append(n_map[d]); t_idx.append(n_map[r]); v_val.append(v)
                
        fig = go.Figure(data=[go.Sankey(node=dict(label=nodes, color="blue"), link=dict(source=s_idx, target=t_idx, value=v_val))])
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        st.subheader("Factory Capacity Utilization (Risk View)")
        util_data = []
        for y in years:
            for f in facs:
                active_cap = sum([pulp.value(build_fac[(f, yb, 'Std')])*df_fac.loc[f, 'Cap_Std'] + pulp.value(build_fac[(f, yb, 'Mega')])*df_fac.loc[f, 'Cap_Mega'] for yb in years if yb <= y])
                if active_cap > 0:
                    flow = sum([pulp.value(p_3pl[(s,f,d,r,y)]) + pulp.value(p_own[(s,f,d,r,y)]) for s in sups for d in dcs for r in regs])
                    util_data.append({"Year": y, "Factory": f, "Throughput": int(flow), "Capacity": int(active_cap), "Utilization %": round((flow/active_cap)*100, 1)})
        if util_data:
            st.dataframe(pd.DataFrame(util_data), column_config={"Utilization %": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100)}, use_container_width=True)
        else:
            st.write("No factories operating.")

    with t4:
        st.subheader("CFO Audit: Exact Multi-Echelon Ledger")
        ledger = []
        for y in years:
            for s in sups:
                for f in facs:
                    for d in dcs:
                        for r in regs:
                            v3 = pulp.value(p_3pl[(s,f,d,r,y)])
                            vo = pulp.value(p_own[(s,f,d,r,y)])
                            total_v = v3 + vo
                            if total_v > 0:
                                # Determine blended handling cost if both paths exist (rare, but mathematically possible)
                                hand_cost = ((v3 * df_3pl.loc[d, 'Variable_Handling_Cost']) + (vo * df_3pl.loc[d, 'Owned_Var_Handling'])) / total_v
                                
                                row = {
                                    "Year": y, "Supplier": s, "Factory": f, "DC": d, "Region": r,
                                    "Units": total_v, "Revenue": total_v * PRICE,
                                    "RM_Tariff": total_v * df_sup.loc[s, 'RM_Cost'] * 1.2,
                                    "Freight_In": total_v * df_f_in.loc[s, f],
                                    "Freight_Out": total_v * df_f_out.loc[f, d],
                                    "DC_Handling": total_v * hand_cost,
                                    "Last_Mile": total_v * df_last.loc[d, r],
                                    "Returns_Leakage": total_v * sim_returns * (df_last.loc[d, r] + sim_refurb)
                                }
                                row["Landed_Cost"] = row["RM_Tariff"] + row["Freight_In"] + row["Freight_Out"] + row["DC_Handling"] + row["Last_Mile"] + row["Returns_Leakage"]
                                row["Contribution_Margin"] = row["Revenue"] - row["Landed_Cost"]
                                row["Margin_%"] = round(row["Contribution_Margin"] / row["Revenue"], 4)
                                ledger.append(row)
                                
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame(ledger).to_excel(writer, sheet_name="Path_Ledger", index=False)
        st.download_button("📥 Download Perfect Traceability Ledger (.xlsx)", data=output.getvalue(), file_name="CFO_Network_Audit.xlsx")

else:
    st.info("👈 Set your strategy and click 'Solve'.")
