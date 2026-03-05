import streamlit as st
import pandas as pd
import pulp
import io
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="UK Strategic Profit Architect", layout="wide")

# --- 1. DATA FACTORY (Restoring Complexity) ---
def generate_rich_data():
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Constants
        pd.DataFrame({"Parameter": ["WACC", "Variable_Cost_Inflation", "Tax Rate"], "Value": [0.10, 0.03, 0.25]}).to_excel(writer, sheet_name="Constants", index=False)
        
        # Suppliers
        pd.DataFrame({"Supplier": ["Shenzhen_China"], "RM_Cost": [750], "Tariff_Rate": [0.20], "Inbound_Multiplier": [1.0]}).to_excel(writer, sheet_name="Suppliers", index=False)
        
        # Facilities (Factories) - DIFFERENT CAPACITIES & COSTS
        pd.DataFrame({
            "Site": ["Manchester", "Birmingham", "Glasgow"],
            "Cap_Std": [40000, 50000, 30000],
            "Cap_Mega": [120000, 150000, 100000],
            "Fixed_Cost_Annual": [800000, 1200000, 600000]
        }).to_excel(writer, sheet_name="Facilities", index=False)
        
        # 3PL / Owned Nodes
        pd.DataFrame({
            "DC_Location": ["North_Hub", "Midlands_Hub", "South_Hub"],
            "Fixed_Cost": [400000, 600000, 800000],
            "Variable_Handling_Cost": [45, 50, 65],
            "Owned_Fixed_Cost": [150000, 250000, 350000],
            "Owned_Var_Handling": [20, 22, 28],
            "Owned_CAPEX": [4e6, 6e6, 8e6]
        }).to_excel(writer, sheet_name="3PL_Nodes", index=False)
        
        # Demand (5 Years)
        years = [1,2,3,4,5]
        regs = ["London", "Scotland", "North_East", "South_West", "NI"]
        demand_matrix = []
        for y in years:
            row = {"Year": y, "London": 15000*y, "Scotland": 6000*y, "North_East": 8000*y, "South_West": 7000*y, "NI": 3000*y}
            demand_matrix.append(row)
        pd.DataFrame(demand_matrix).to_excel(writer, sheet_name="Demand", index=False)
        
        # Freight Matrix (Unique for every pair)
        pd.DataFrame([{"From": "Shenzhen_China", "Manchester": 180, "Birmingham": 150, "Glasgow": 210}]).to_excel(writer, sheet_name="Freight_Inbound", index=False)
        
        fo_data = [
            {"From": "Manchester", "North_Hub": 15, "Midlands_Hub": 45, "South_Hub": 85},
            {"From": "Birmingham", "North_Hub": 45, "Midlands_Hub": 15, "South_Hub": 45},
            {"From": "Glasgow", "North_Hub": 65, "Midlands_Hub": 95, "South_Hub": 150}
        ]
        pd.DataFrame(fo_data).to_excel(writer, sheet_name="Freight_Outbound", index=False)
        
        lm_data = [
            {"From": "North_Hub", "London": 120, "Scotland": 80, "North_East": 25, "South_West": 140, "NI": 250},
            {"From": "Midlands_Hub", "London": 70, "Scotland": 140, "North_East": 60, "South_West": 80, "NI": 220},
            {"From": "South_Hub", "London": 25, "Scotland": 250, "North_East": 130, "South_West": 50, "NI": 350}
        ]
        pd.DataFrame(lm_data).to_excel(writer, sheet_name="Last_Mile", index=False)
        
    return output.getvalue()

# --- 2. HEADER & OBJECTIVE ---
st.title("🇬🇧 UK Strategic Network: SPEC-Profit Architect")
with st.expander("📌 Optimization Objective & Commercial Logic", expanded=False):
    st.markdown("""
    **Objective:** Maximize **5-Year NPV** by optimizing infrastructure builds and commercial footprint.
    
    **The Solver Specifics:**
    * **Unit SPEC:** Every Factory/DC/Region pair has a unique cost.
    * **Rationalization:** In 'Profit Max' mode, the solver exits a region if `Wholesale Revenue < COGS + Double-Freight + Refurbishment`.
    * **Volume Flows:** The Sankey Diagram traces **Physical Units** to visualize network density.
    """)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("📥 Data Management")
    st.download_button("📥 Download Complex Template", data=generate_rich_data(), file_name="Strategic_Network_Master.xlsx")
    uploaded_file = st.file_uploader("Upload Network Spec (.xlsx)", type=["xlsx"])

    st.header("🏢 Strategy & Policy")
    strategy = st.radio("Asset Strategy", ["Optimize Mix", "3PL Only", "Owned Only"])
    service_mode = st.radio("Commercial Policy", ["Capture All Demand", "Profit Max (Rationalize)"])
    
    st.header("🔄 Reverse Logistics")
    sim_returns = st.slider("Global Returns (%)", 0, 30, 8) / 100.0
    sim_refurb = st.slider("Refurb Cost (£/unit)", 50, 500, 150)

    run_button = st.button("🚀 Solve Strategic Network", type="primary", use_container_width=True)

# --- 4. THE SOLVER ---
if run_button:
    file_name = uploaded_file if uploaded_file else io.BytesIO(generate_rich_data())
    try:
        # Full echelon load
        df_fac = pd.read_excel(file_name, sheet_name='Facilities').set_index('Site')
        df_dem = pd.read_excel(file_name, sheet_name='Demand').set_index('Year')
        df_sup = pd.read_excel(file_name, sheet_name='Suppliers').set_index('Supplier')
        df_3pl = pd.read_excel(file_name, sheet_name='3PL_Nodes').set_index('DC_Location')
        df_f_in = pd.read_excel(file_name, sheet_name='Freight_Inbound').set_index('From')
        df_f_out = pd.read_excel(file_name, sheet_name='Freight_Outbound').set_index('From')
        df_last = pd.read_excel(file_name, sheet_name='Last_Mile').set_index('From')
        df_const = pd.read_excel(file_name, sheet_name='Constants').set_index('Parameter')
    except Exception as e:
        st.error(f"Excel Load Error: {e}")
        st.stop()

    WACC = df_const.loc['WACC', 'Value']
    PRICE = 2500
    years, sups, facs, dcs, regs = [1,2,3,4,5], df_sup.index.tolist(), df_fac.index.tolist(), df_3pl.index.tolist(), df_dem.columns.tolist()

    model = pulp.LpProblem("UK_Strategic_Network", pulp.LpMaximize)

    # Variables (Fixed Indexing)
    build_fac = pulp.LpVariable.dicts("BuildFac", ((f, y, s) for f in facs for y in years for s in ['Std', 'Mega']), cat='Binary')
    open_3pl = pulp.LpVariable.dicts("Open3PL", ((d, y) for d in dcs for y in years), cat='Binary')
    build_own = pulp.LpVariable.dicts("BuildOwn", ((d, y) for d in dcs for y in years), cat='Binary')

    f_in = pulp.LpVariable.dicts("In", ((s, f, y) for s in sups for f in facs for y in years), lowBound=0)
    f_out = pulp.LpVariable.dicts("Out", ((f, d, y) for f in facs for d in dcs for y in years), lowBound=0)
    f_last = pulp.LpVariable.dicts("Last", ((d, r, y) for d in dcs for r in regs for y in years), lowBound=0)

    # Constraints
    for y in years:
        for r in regs:
            total_served = pulp.lpSum([f_last[(d, r, y)] for d in dcs])
            if service_mode == "Capture All Demand": model += total_served == df_dem.loc[y, r]
            else: model += total_served <= df_dem.loc[y, r]

        for d in dcs:
            model += pulp.lpSum([f_out[(f, d, y)] for f in facs]) == pulp.lpSum([f_last[(d, r, y)] for r in regs])
            if strategy == "3PL Only": model += build_own[(d, y)] == 0
            elif strategy == "Owned Only": model += open_3pl[(d, y)] == 0
            model += pulp.lpSum([f_last[(d, r, y)] for r in regs]) <= (open_3pl[(d, y)] + pulp.lpSum([build_own[(d, yb)] for yb in years if yb <= y])) * 1000000

        for f in facs:
            model += pulp.lpSum([f_in[(s, f, y)] for s in sups]) == pulp.lpSum([f_out[(f, d, y)] for d in dcs])
            cap = pulp.lpSum([build_fac[(f, yb, 'Std')] * df_fac.loc[f, 'Cap_Std'] + build_fac[(f, yb, 'Mega')] * df_fac.loc[f, 'Cap_Mega'] for yb in years if yb <= y])
            model += pulp.lpSum([f_out[(f, d, y)] for d in dcs]) <= cap

    # NPV Calculation
    discounted_cfs = []
    for y in years:
        rev = pulp.lpSum([f_last[(d, r, y)] * PRICE for d in dcs for r in regs])
        c_prod = pulp.lpSum([f_in[(s, f, y)] * (df_sup.loc[s, 'RM_Cost'] * (1 + df_sup.loc[s, 'Tariff_Rate']) + df_f_in.loc[s, f]) for s in sups for f in facs])
        c_fwd = pulp.lpSum([f_out[(f, d, y)] * df_f_out.loc[f, d] for f in facs for d in dcs]) + \
                pulp.lpSum([f_last[(d, r, y)] * (df_last.loc[d, r] + 50) for d in dcs for r in regs])
        c_rev = pulp.lpSum([f_last[(d, r, y)] * sim_returns * (df_last.loc[d, r] + sim_refurb) for d in dcs for r in regs])
        fixed = pulp.lpSum([open_3pl[(d, y)] * df_3pl.loc[d, 'Fixed_Cost'] + build_own[(d, y)] * df_3pl.loc[d, 'Owned_Fixed_Cost'] for d in dcs])
        capex = pulp.lpSum([build_fac[(f, y, sz)] * 5e6 for f in facs for sz in ['Std', 'Mega']])
        discounted_cfs.append((rev - c_prod - c_fwd - c_rev - fixed - capex) / ((1+WACC)**y))

    model += pulp.lpSum(discounted_cfs)
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # --- 5. UI OUTPUTS ---
    t1, t2, t3, t4 = st.tabs(["🏗️ Optimal Infrastructure", "🚚 Volume Flow (Units)", "📍 Regional Profitability", "📥 CFO Audit Ledger"])

    with t1:
        st.subheader("5-Year Strategic Build Schedule")
        build_log = []
        for y in years:
            for f in facs:
                for sz in ['Std', 'Mega']:
                    if pulp.value(build_fac[(f, y, sz)]) > 0.5: build_log.append({"Year": y, "Asset": f, "Type": f"Factory ({sz})"})
            for d in dcs:
                if pulp.value(build_own[(d, y)]) > 0.5: build_log.append({"Year": y, "Asset": d, "Type": "Owned DC"})
        if build_log: st.table(pd.DataFrame(build_log))
        else: st.info("Asset-Light strategy chosen: No CAPEX infrastructure built.")

    with t2:
        st.subheader("Sankey Diagram: Product Volume (Units)")
        # Nodes list for Sankey
        all_nodes = sups + facs + dcs + regs
        node_map = {n: i for i, n in enumerate(all_nodes)}
        src, tgt, val = [], [], []
        # Sup to Fac
        for s in sups:
            for f in facs:
                v = sum([pulp.value(f_in[(s,f,y)]) for y in years])
                if v > 1: src.append(node_map[s]); tgt.append(node_map[f]); val.append(v)
        # Fac to DC
        for f in facs:
            for d in dcs:
                v = sum([pulp.value(f_out[(f,d,y)]) for y in years])
                if v > 1: src.append(node_map[f]); tgt.append(node_map[d]); val.append(v)
        # DC to Reg
        for d in dcs:
            for r in regs:
                v = sum([pulp.value(f_last[(d,r,y)]) for y in years])
                if v > 1: src.append(node_map[d]); tgt.append(node_map[r]); val.append(v)

        fig = go.Figure(data=[go.Sankey(
            node = dict(label = all_nodes, color = "blue"),
            link = dict(source = src, target = tgt, value = val))])
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.caption("Visualizing the 5-Year physical throughput. Thicker lines indicate higher logistics density.")

    with t3:
        st.subheader("Regional Contribution Margin % (Rationalization View)")
        reg_profit = []
        for r in regs:
            vol = sum([pulp.value(f_last[(d,r,y)]) for d in dcs for y in years])
            if vol > 0:
                rev = vol * PRICE
                cost = sum([pulp.value(f_last[(d,r,y)]) * (1200 + df_last.loc[d,r] + (sim_returns * (df_last.loc[d,r] + sim_refurb))) for d in dcs for y in years])
                reg_profit.append({"Region": r, "Total Units": int(vol), "CM %": round((rev-cost)/rev*100, 1), "Net Profit": rev-cost})
        df_p = pd.DataFrame(reg_profit).sort_values("CM %", ascending=False)
        st.dataframe(df_p.style.background_gradient(subset=['CM %'], cmap='RdYlGn'), use_container_width=True)

    with t4:
        st.subheader("CFO Data Export")
        audit = []
        for y in years:
            for d in dcs:
                for r in regs:
                    v = pulp.value(f_last[(d,r,y)])
                    if v > 0: audit.append({"Year": y, "From_DC": d, "To_Region": r, "Units": v, "Logistics_Cost": v*df_last.loc[d,r]})
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame(audit).to_excel(writer, sheet_name="Transactional_Ledger", index=False)
            df_p.to_excel(writer, sheet_name="Regional_Profitability", index=False)
        st.download_button("📥 Download Financial Audit (.xlsx)", data=output.getvalue(), file_name="Network_Optimization_Audit.xlsx")

else:
    st.info("👈 Open the sidebar, download the complex template, and click 'Solve' to run the SPEC-grade architect.")
