import streamlit as st
import pandas as pd
import pulp
import math
import plotly.graph_objects as go

st.set_page_config(page_title="UK Supply Chain Architect", layout="wide")

# --- 1. PROBLEM STATEMENT / OBJECTIVE ---
st.title("🇬🇧 UK Strategic Network Design Architect")
with st.expander("📌 Strategic Objective & Optimization Logic", expanded=True):
    st.markdown("""
    **Objective:** Maximize the **5-Year Net Present Value (NPV)** of the UK business unit.
    
    **The Optimization Challenge:**
    * **Multi-Echelon Flow:** Supplier -> Factory (Std/Mega) -> DC (Owned/3PL) -> Customer Region.
    * **Reverse Logistics:** Customer Region -> DC (Returns processing and refurbishment).
    * **Financial Decision:** The model weighs the **CAPEX** of building owned infrastructure (high upfront cost, low variable handling) against the **Asset-Light** 3PL approach (zero upfront cost, high variable fees).
    """)

# --- 2. SIDEBAR SCENARIO CONTROLS ---
with st.sidebar:
    st.header("🌍 Global Levers")
    sim_tariff = st.slider("China Tariff Rate (%)", 0.0, 0.5, 0.20)
    sim_freight = st.slider("Ocean Freight (China -> UK)", 50, 500, 150)
    
    st.header("📈 Market & Returns")
    sim_demand = st.slider("Demand Growth Multiplier", 0.5, 2.0, 1.0)
    sim_returns = st.slider("Returns Rate (%)", 0, 25, 8) / 100.0
    sim_refurb_cost = st.number_input("Refurbishment Cost (£/unit)", 50, 500, 150)
    
    st.header("🏗️ Corporate Strategy")
    strategy = st.radio("Distribution Strategy", [
        "Optimize Mix (Owned + 3PL)", 
        "100% Asset-Light (3PL Only)", 
        "100% Asset-Heavy (Owned Only)"
    ])
    
    run_button = st.button("🚀 Run Strategic Solver", type="primary", use_container_width=True)

if run_button:
    # 1. DATA INGESTION
    file_name = 'SC_Model_Data.xlsx'
    try:
        df_fac = pd.read_excel(file_name, sheet_name='Facilities').set_index('Site')
        df_const = pd.read_excel(file_name, sheet_name='Constants').set_index('Parameter')
        df_dem = pd.read_excel(file_name, sheet_name='Demand').set_index('Year')
        df_sup = pd.read_excel(file_name, sheet_name='Suppliers').set_index('Supplier')
        df_3pl = pd.read_excel(file_name, sheet_name='3PL_Nodes').set_index('DC_Location')
        df_freight_in = pd.read_excel(file_name, sheet_name='Freight_Inbound').set_index('From')
        df_freight_out = pd.read_excel(file_name, sheet_name='Freight_Outbound').set_index('From')
        df_last_mile = pd.read_excel(file_name, sheet_name='Last_Mile').set_index('From')
    except Exception as e:
        st.error(f"Excel Error: {e}")
        st.stop()

    # 2. CONSTANTS
    WACC = df_const.loc['WACC', 'Value']
    INFLATION = df_const.loc['Variable_Cost_Inflation', 'Value']
    PRICE = 2500
    TAX_RATE = 0.25
    years, sups, facs, dcs, regs = [1,2,3,4,5], df_sup.index.tolist(), df_fac.index.tolist(), df_3pl.index.tolist(), df_dem.columns.tolist()

    # 3. MILP SETUP
    model = pulp.LpProblem("UK_Strategic_Network", pulp.LpMaximize)

    # Binaries
    build_fac = pulp.LpVariable.dicts("BuildFac", ((f, y, s) for f in facs for y in years for s in ['Std', 'Mega']), cat='Binary')
    open_3pl = pulp.LpVariable.dicts("Open3PL", ((dc, y) for dc in dcs for y in years), cat='Binary')
    build_own_dc = pulp.LpVariable.dicts("BuildOwnDC", ((dc, y) for dc in dcs for y in years), cat='Binary')

    # Flows
    f_in = pulp.LpVariable.dicts("FlowIn", ((s, f, y) for s in sups for f in facs for y in years), lowBound=0)
    f_out = pulp.LpVariable.dicts("FlowOut", ((f, dc, y) for f in facs for dc in dcs for y in years), lowBound=0)
    f_last_3pl = pulp.LpVariable.dicts("Flow3PL", ((dc, r, y) for dc in dcs for r in regs for y in years), lowBound=0)
    f_last_own = pulp.LpVariable.dicts("FlowOwn", ((dc, r, y) for dc in dcs for r in regs for y in years), lowBound=0)

    BIG_M = 1000000

    # 4. CONSTRAINTS
    for y in years:
        for r in regs:
            model += pulp.lpSum([f_last_3pl[dc, r, y] + f_last_own[dc, r, y] for dc in dcs]) == df_dem.loc[y, r] * sim_demand
        
        for f in facs:
            model += pulp.lpSum([f_in[s, f, y] for s in sups]) == pulp.lpSum([f_out[f, dc, y] for dc in dcs])
            cap = pulp.lpSum([build_fac[f, yb, 'Std'] * df_fac.loc[f, 'Cap_Std'] + build_fac[f, yb, 'Mega'] * df_fac.loc[f, 'Cap_Mega'] for yb in years if yb <= y])
            model += pulp.lpSum([f_out[f, dc, y] for dc in dcs]) <= cap

        for dc in dcs:
            model += pulp.lpSum([f_out[f, dc, y] for f in facs]) == pulp.lpSum([f_last_3pl[dc, r, y] + f_last_own[dc, r, y] for r in regs])
            if strategy == "100% Asset-Light (3PL Only)": model += build_own_dc[dc, y] == 0
            elif strategy == "100% Asset-Heavy (Owned Only)": model += open_3pl[dc, y] == 0
            model += pulp.lpSum([f_last_3pl[dc, r, y] for r in regs]) <= open_3pl[dc, y] * BIG_M
            model += pulp.lpSum([f_last_own[dc, r, y] for r in regs]) <= pulp.lpSum([build_own_dc[dc, yb] for yb in years if yb <= y]) * BIG_M

    # 5. NPV CALCULATION
    cashflows = []
    for y in years:
        inf = (1 + INFLATION)**(y - 1)
        rev = pulp.lpSum([(f_last_3pl[dc, r, y] + f_last_own[dc, r, y]) * PRICE for dc in dcs for r in regs])
        c_in = pulp.lpSum([f_in[s, f, y] * (df_sup.loc[s, 'RM_Cost'] * (1 + sim_tariff) + sim_freight) for s in sups for f in facs])
        c_out = pulp.lpSum([f_out[f, dc, y] * df_freight_out.loc[f, dc] for f in facs for dc in dcs])
        c_lm = pulp.lpSum([(f_last_3pl[dc, r, y] + f_last_own[dc, r, y]) * df_last_mile.loc[dc, r] for dc in dcs for r in regs])
        c_hand = pulp.lpSum([f_last_3pl[dc, r, y] * df_3pl.loc[dc, 'Variable_Handling_Cost'] + f_last_own[dc, r, y] * df_3pl.loc[dc, 'Owned_Var_Handling'] for dc in dcs for r in regs])
        
        # REVERSE LOGISTICS
        c_rev = pulp.lpSum([(f_last_3pl[dc, r, y] + f_last_own[dc, r, y]) * sim_returns * (df_last_mile.loc[dc, r] + sim_refurb_cost) for dc in dcs for r in regs])
        
        fixed = pulp.lpSum([open_3pl[dc, y] * df_3pl.loc[dc, 'Fixed_Cost'] + build_own_dc[dc, y] * df_3pl.loc[dc, 'Owned_Fixed_Cost'] for dc in dcs])
        capex = pulp.lpSum([build_fac[f, y, 'Std'] * 5000000 + build_fac[f, y, 'Mega'] * 12000000 for f in facs]) + \
                pulp.lpSum([build_own_dc[dc, y] * 3000000 for dc in dcs])

        cashflows.append((rev - (c_in + c_out + c_lm + c_hand + c_rev + fixed + capex)) / ((1 + WACC)**y))

    model += pulp.lpSum(cashflows)
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # 6. RESULTS & DASHBOARD
    t_rev = sum([pulp.value(f_last_3pl[dc, r, y] + f_last_own[dc, r, y]) * PRICE for dc in dcs for r in regs for y in years])
    t_in = sum([pulp.value(f_in[s, f, y]) * (df_sup.loc[s, 'RM_Cost'] * (1 + sim_tariff) + sim_freight) for s in sups for f in facs for y in years])
    t_log = sum([pulp.value(f_out[f, dc, y] * df_freight_out.loc[f, dc] + (f_last_3pl[dc, r, y] + f_last_own[dc, r, y]) * df_last_mile.loc[dc, r]) for f in facs for dc in dcs for r in regs for y in years])
    t_rev_log = sum([pulp.value(f_last_3pl[dc, r, y] + f_last_own[dc, r, y]) * sim_returns * (df_last_mile.loc[dc, r] + sim_refurb_cost) for dc in dcs for r in regs for y in years])
    t_fixed = sum([pulp.value(open_3pl[dc, y] * 500000 + build_own_dc[dc, y] * 200000) for dc in dcs for y in years])
    t_capex = sum([pulp.value(build_fac[f, y, sz] * 5000000) for f in facs for y in years for sz in ['Std', 'Mega']])

    tab1, tab2, tab3 = st.tabs(["🚀 Executive Summary", "💰 5-Year P&L", "🚚 Logistics Map"])

    with tab1:
        st.subheader("NPV Value Bridge: Revenue to Net Cash Flow")
        fig = go.Figure(go.Waterfall(
            orientation = "v", measure = ["absolute", "relative", "relative", "relative", "relative", "relative", "total"],
            x = ["Revenue", "COGS (Inbound)", "Logistics (Fwd)", "Reverse Logistics", "Fixed Costs", "CAPEX", "Net Cash Flow"],
            y = [t_rev, -t_in, -t_log, -t_rev_log, -t_fixed, -t_capex, 0],
            text = [f"£{v/1e6:.1f}M" for v in [t_rev, -t_in, -t_log, -t_rev_log, -t_fixed, -t_capex]],
            decreasing = {"marker":{"color":"#FF4B4B"}}, increasing = {"marker":{"color":"#2ca02c"}}, totals = {"marker":{"color":"#1f77b4"}}
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Project NPV (WACC 10%)", f"£{pulp.value(model.objective)/1e6:.1f}M")
        c2.metric("EBITDA Margin", f"{(t_rev-t_in-t_log-t_rev_log-t_fixed)/t_rev*100:.1f}%")
        c3.metric("Reverse Logistics Leakage", f"£{t_rev_log/1e6:.1f}M")

    with tab2:
        st.subheader("Detailed Cumulative Financials")
        st.table(pd.DataFrame({
            "Line Item": ["Gross Revenue", "Inbound COGS (Tariffs & Freight)", "Forward Logistics", "Reverse Logistics (Returns)", "Infrastructure Fixed Costs", "Growth CAPEX"],
            "Amount (£M)": [t_rev/1e6, -t_in/1e6, -t_log/1e6, -t_rev_log/1e6, -t_fixed/1e6, -t_capex/1e6]
        }))

    with tab3:
        st.subheader("Optimal Strategy Footprint")
        st.write("**Infrastructure Built/Opened:**")
        sites = []
        for y in years:
            for f in facs:
                if build_fac[f,y,'Std'].varValue or build_fac[f,y,'Mega'].varValue: sites.append({"Year": y, "Site": f, "Type": "Factory"})
            for dc in dcs:
                if open_3pl[dc,y].varValue: sites.append({"Year": y, "DC": dc, "Type": "3PL DC"})
                if build_own_dc[dc,y].varValue: sites.append({"Year": y, "DC": dc, "Type": "Owned DC"})
        st.dataframe(pd.DataFrame(sites))
else:
    st.info("👈 Use the sidebar to set global macro levers and corporate strategy, then click 'Run Strategic Solver'.")
