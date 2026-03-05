\import streamlit as st
import pandas as pd
import pulp
import io
import plotly.graph_objects as go

st.set_page_config(page_title="UK Strategic Network: CFO Audit Architect", layout="wide")

# --- 1. DATA FACTORY: RESTORING HETEROGENEITY ---
def generate_original_complex_data():
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Constants
        pd.DataFrame({"Parameter": ["WACC", "Variable_Cost_Inflation", "Tax Rate"], "Value": [0.10, 0.03, 0.25]}).to_excel(writer, sheet_name="Constants", index=False)
        
        # Suppliers (China Base)
        pd.DataFrame({"Supplier": ["Shenzhen_China"], "RM_Cost": [750], "Tariff_Rate": [0.20], "Inbound_Multiplier": [1.0]}).to_excel(writer, sheet_name="Suppliers", index=False)
        
        # Facilities (Factories) - DIFFERENT CAPACITIES & FIXED COSTS
        pd.DataFrame({
            "Site": ["Manchester", "Birmingham", "Glasgow"],
            "Cap_Std": [45000, 60000, 35000],
            "Cap_Mega": [130000, 180000, 110000],
            "Fixed_Cost_Annual": [900000, 1400000, 700000]
        }).to_excel(writer, sheet_name="Facilities", index=False)
        
        # Distribution Nodes (3PL vs Owned)
        pd.DataFrame({
            "DC_Location": ["North_Hub", "Midlands_Hub", "South_Hub"],
            "Fixed_Cost": [400000, 650000, 800000],
            "Variable_Handling_Cost": [45, 50, 65],
            "Owned_Fixed_Cost": [180000, 280000, 380000],
            "Owned_Var_Handling": [20, 22, 28],
            "Owned_CAPEX": [4e6, 6e6, 8e6]
        }).to_excel(writer, sheet_name="3PL_Nodes", index=False)
        
        # Demand (5 Years) - ASYMMETRIC REGIONS
        years, regs = [1,2,3,4,5], ["London", "Scotland", "North_East", "South_West", "NI"]
        demand_matrix = [{"Year": y, "London": 20000*y, "Scotland": 8000*y, "North_East": 10000*y, "South_West": 9000*y, "NI": 4000*y} for y in years]
        pd.DataFrame(demand_matrix).to_excel(writer, sheet_name="Demand", index=False)
        
        # Freight Matrices (THE CORE OF OPTIMIZATION)
        pd.DataFrame([{"From": "Shenzhen_China", "Manchester": 180, "Birmingham": 150, "Glasgow": 220}]).to_excel(writer, sheet_name="Freight_Inbound", index=False)
        
        pd.DataFrame([
            {"From": "Manchester", "North_Hub": 12, "Midlands_Hub": 48, "South_Hub": 95},
            {"From": "Birmingham", "North_Hub": 48, "Midlands_Hub": 15, "South_Hub": 45},
            {"From": "Glasgow", "North_Hub": 65, "Midlands_Hub": 110, "South_Hub": 180}
        ]).to_excel(writer, sheet_name="Freight_Outbound", index=False)
        
        pd.DataFrame([
            {"From": "North_Hub", "London": 130, "Scotland": 75, "North_East": 25, "South_West": 150, "NI": 260},
            {"From": "Midlands_Hub", "London": 70, "Scotland": 150, "North_East": 65, "South_West": 85, "NI": 230},
            {"From": "South_Hub", "London": 25, "Scotland": 280, "North_East": 140, "South_West": 55, "NI": 380}
        ]).to_excel(writer, sheet_name="Last_Mile", index=False)
        
    return output.getvalue()

# --- 2. UI & OBJECTIVE ---
st.title("🇬🇧 UK Strategic Network: CFO Profit Architect")
with st.expander("📌 Strategic Problem Statement", expanded=True):
    st.markdown("""
    **Problem:** The current distribution footprint is static. Rising last-mile costs and reverse logistics are eroding margins in specific regions.
    **Objective:** Use a multi-echelon MILP to find the 5-Year NPV ceiling. 
    **Key Lever:** *Profit Max Mode* allows the solver to exit regions where the 'Cost-to-Serve' (Forward + Reverse Logistics) exceeds the unit margin.
    """)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("📥 Data Management")
    st.download_button("📥 Download Rich Spec Template", data=generate_original_complex_data(), file_name="Network_Profit_Spec.xlsx")
    uploaded_file = st.file_uploader("Upload Network Data (.xlsx)", type=["xlsx"])
    st.header("🏢 Asset Policy")
    strategy = st.radio("Strategy", ["Optimize Mix", "3PL Only", "Owned Only"])
    service_mode = st.radio("Service Policy", ["Capture All Demand", "Profit Max (Rationalize)"])
    st.header("🔄 Reverse Logistics")
    sim_returns = st.slider("Returns Rate (%)", 0, 30, 10) / 100.0
    sim_refurb = st.slider("Refurb Cost (£/unit)", 50, 500, 150)
    run_button = st.button("🚀 Run Profit Optimizer", type="primary", use_container_width=True)

# --- 4. SOLVER ---
if run_button:
    source = uploaded_file if uploaded_file else io.BytesIO(generate_original_complex_data())
    try:
        df_fac = pd.read_excel(source, sheet_name='Facilities').set_index('Site')
        df_dem = pd.read_excel(source, sheet_name='Demand').set_index('Year')
        df_sup = pd.read_excel(source, sheet_name='Suppliers').set_index('Supplier')
        df_3pl = pd.read_excel(source, sheet_name='3PL_Nodes').set_index('DC_Location')
        df_f_in = pd.read_excel(source, sheet_name='Freight_Inbound').set_index('From')
        df_f_out = pd.read_excel(source, sheet_name='Freight_Outbound').set_index('From')
        df_last = pd.read_excel(source, sheet_name='Last_Mile').set_index('From')
        df_const = pd.read_excel(source, sheet_name='Constants').set_index('Parameter')
    except Exception as e:
        st.error(f"Schema Error: {e}")
        st.stop()

    WACC, PRICE = df_const.loc['WACC', 'Value'], 2500
    years, sups, facs, dcs, regs = [1,2,3,4,5], df_sup.index.tolist(), df_fac.index.tolist(), df_3pl.index.tolist(), df_dem.columns.tolist()

    model = pulp.LpProblem("UK_Strategic_Network", pulp.LpMaximize)

    # Variables
    build_fac = pulp.LpVariable.dicts("BF", ((f, y, s) for f in facs for y in years for s in ['Std', 'Mega']), cat='Binary')
    build_own = pulp.LpVariable.dicts("BO", ((d, y) for d in dcs for y in years), cat='Binary')
    open_3pl = pulp.LpVariable.dicts("O3", ((d, y) for d in dcs for y in years), cat='Binary')
    f_in = pulp.LpVariable.dicts("In", ((s, f, y) for s in sups for f in facs for y in years), lowBound=0)
    f_out = pulp.LpVariable.dicts("Out", ((f, d, y) for f in facs for d in dcs for y in years), lowBound=0)
    f_last = pulp.LpVariable.dicts("Last", ((d, r, y) for d in dcs for r in regs for y in years), lowBound=0)

    # Constraints
    for y in years:
        for r in regs:
            vol = pulp.lpSum([f_last[(d, r, y)] for d in dcs])
            if service_mode == "Capture All Demand": model += vol == df_dem.loc[y, r]
            else: model += vol <= df_dem.loc[y, r]
        for d in dcs:
            model += pulp.lpSum([f_out[(f, d, y)] for f in facs]) == pulp.lpSum([f_last[(d, r, y)] for r in regs])
            if strategy == "3PL Only": model += build_own[(d, y)] == 0
            elif strategy == "Owned Only": model += open_3pl[(d, y)] == 0
            model += pulp.lpSum([f_last[(d, r, y)] for r in regs]) <= (open_3pl[(d, y)] + pulp.lpSum([build_own[(d, yb)] for yb in years if yb <= y])) * 1e7
        for f in facs:
            model += pulp.lpSum([f_in[(s, f, y)] for s in sups]) == pulp.lpSum([f_out[(f, d, y)] for d in dcs])
            cap = pulp.lpSum([build_fac[(f, yb, 'Std')] * df_fac.loc[f, 'Cap_Std'] + build_fac[(f, yb, 'Mega')] * df_fac.loc[f, 'Cap_Mega'] for yb in years if yb <= y])
            model += pulp.lpSum([f_out[(f, d, y)] for d in dcs]) <= cap

    # Objectives
    cfs = []
    for y in years:
        rev = pulp.lpSum([f_last[(d, r, y)] * PRICE for d in dcs for r in regs])
        c_prod = pulp.lpSum([f_in[(s, f, y)] * (df_sup.loc[s, 'RM_Cost']*1.2 + df_f_in.loc[s, f]) for s in sups for f in facs])
        c_fwd = pulp.lpSum([f_out[(f, d, y)] * df_f_out.loc[f, d] for f in facs for d in dcs]) + \
                pulp.lpSum([f_last[(d, r, y)] * (df_last.loc[d, r] + df_3pl.loc[d, 'Variable_Handling_Cost']) for d in dcs for r in regs])
        c_rev = pulp.lpSum([f_last[(d, r, y)] * sim_returns * (df_last.loc[d, r] + sim_refurb) for d in dcs for r in regs])
        fixed = pulp.lpSum([open_3pl[(d, y)] * df_3pl.loc[d, 'Fixed_Cost'] + build_own[(d, y)] * df_3pl.loc[d, 'Owned_Fixed_Cost'] for d in dcs]) + \
                pulp.lpSum([build_fac[(f, y, sz)] * df_fac.loc[f, 'Fixed_Cost_Annual'] for f in facs for sz in ['Std', 'Mega']])
        capex = pulp.lpSum([build_own[(d, y)] * df_3pl.loc[d, 'Owned_CAPEX'] for d in dcs]) + pulp.lpSum([build_fac[(f, y, sz)] * 5e6 for f in facs for sz in ['Std', 'Mega']])
        cfs.append((rev - c_prod - c_fwd - c_rev - fixed - capex) / ((1 + WACC)**y))

    model += pulp.lpSum(cfs)
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # --- 5. RESULTS ---
    t1, t2, t3, t4 = st.tabs(["🏗️ Infrastructure Audit", "🚚 Volume Flow (Sankey)", "📍 Regional Profitability", "📥 Detailed CFO Ledger"])

    with t1:
        st.subheader("Asset Build Schedule")
        builds = [{"Year": y, "Location": a, "Type": "Factory" if a in facs else "Owned DC"} for y in years for a in facs + dcs if (a in facs and any(pulp.value(build_fac[(a, y, sz)]) > 0.5 for sz in ['Std', 'Mega'])) or (a in dcs and pulp.value(build_own[(a, y)]) > 0.5)]
        st.table(pd.DataFrame(builds) if builds else pd.DataFrame([{"Status": "Asset-Light Solution Chosen"}]))

    with t2:
        st.subheader("Physical Unit Journey (5-Year Throughput)")
        nodes = sups + facs + dcs + regs
        n_map = {n: i for i, n in enumerate(nodes)}
        s_idx, t_idx, v_val = [], [], []
        for s in sups:
            for f in facs:
                v = sum([pulp.value(f_in[(s,f,y)]) for y in years])
                if v > 1: s_idx.append(n_map[s]); t_idx.append(n_map[f]); v_val.append(v)
        for f in facs:
            for d in dcs:
                v = sum([pulp.value(f_out[(f,d,y)]) for y in years])
                if v > 1: s_idx.append(n_map[f]); t_idx.append(n_map[d]); v_val.append(v)
        for d in dcs:
            for r in regs:
                v = sum([pulp.value(f_last[(d,r,y)]) for y in years])
                if v > 1: s_idx.append(n_map[d]); t_idx.append(n_map[r]); v_val.append(v)
        fig = go.Figure(data=[go.Sankey(node=dict(label=nodes, color="blue"), link=dict(source=s_idx, target=t_idx, value=v_val))])
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        st.subheader("Verification of Lane Profitability")
        reg_stats = []
        for r in regs:
            vol = sum([pulp.value(f_last[(d,r,y)]) for d in dcs for y in years])
            if vol > 0:
                t_rev = vol * PRICE
                t_cost = 0
                for y in years:
                    for d in dcs:
                        v = pulp.value(f_last[(d,r,y)])
                        if v > 0:
                            # LANE-SPECIFIC COST TRACING
                            t_cost += v * (df_sup.iloc[0]['RM_Cost']*1.2 + df_last.loc[d,r] + df_3pl.loc[d,'Variable_Handling_Cost'] + sim_returns*(df_last.loc[d,r] + sim_refurb))
                reg_stats.append({"Region": r, "Units": int(vol), "Contribution Margin %": round((t_rev-t_cost)/t_rev*100, 1), "Net CM (£)": t_rev-t_cost})
        df_p = pd.DataFrame(reg_stats).sort_values("Contribution Margin %", ascending=False)
        st.dataframe(df_p, column_config={"Contribution Margin %": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100)}, use_container_width=True)

    with t4:
        st.subheader("Transactional Audit Download")
        ledger = []
        for y in years:
            for d in dcs:
                for r in regs:
                    v = pulp.value(f_last[(d,r,y)])
                    if v > 0:
                        ledger.append({
                            "Year": y, "Region": r, "Source_DC": d, "Units": v,
                            "Unit_Price": PRICE,
                            "Landed_COGS_Est": df_sup.iloc[0]['RM_Cost']*1.2,
                            "Forward_Last_Mile": df_last.loc[d,r],
                            "DC_Handling": df_3pl.loc[d,'Variable_Handling_Cost'],
                            "Reverse_Logistics_Leakage": sim_returns * (df_last.loc[d,r] + sim_refurb),
                            "Total_Landed_Cost": (df_sup.iloc[0]['RM_Cost']*1.2 + df_last.loc[d,r] + df_3pl.loc[d,'Variable_Handling_Cost'] + sim_returns*(df_last.loc[d,r]+sim_refurb))
                        })
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame(ledger).to_excel(writer, sheet_name="Audit_Ledger", index=False)
        st.download_button("📥 Download Granular CFO Audit (.xlsx)", data=output.getvalue(), file_name="UK_Network_CFO_Audit.xlsx")

else:
    st.info("👈 Configure your strategy and click 'Solve'. Download the Rich Template to see the asymmetric costs.")
