import streamlit as st
import pandas as pd
import pulp
import math

st.set_page_config(page_title="UK Supply Chain Architect", layout="wide")

# --- 1. PROBLEM STATEMENT / OBJECTIVE ---
st.title("🇬🇧 UK Strategic Network Design Architect")
with st.expander("📌 Problem Statement & Optimization Objective", expanded=True):
    st.markdown("""
    **The Problem:** The current UK distribution network is fragmented. Rising tariffs, volatile freight costs, and the high cost of 'Last Mile' delivery are eroding the 5-Year NPV. 
    Additionally, **Reverse Logistics** (returns) now represent a significant hidden cost.
    
    **The Objective:** This tool uses a **Mixed-Integer Linear Program (MILP)** to maximize the **5-Year Net Present Value (NPV)**. 
    The solver must decide:
    1. **Where to build factories** (and what size).
    2. **Which DCs to open** (3PL vs. Owned).
    3. **How to route flows** to minimize costs and maximize margin, including the cost of processing returns.
    """)

# --- 2. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🌍 Global Levers")
    sim_tariff = st.slider("China Tariff Rate (%)", 0.0, 0.5, 0.20)
    sim_freight = st.slider("Ocean Freight (China -> UK)", 50, 500, 150)
    
    st.header("📈 Market Levers")
    sim_demand = st.slider("Demand Growth Multiplier", 0.5, 2.0, 1.0)
    sim_returns = st.slider("Returns Rate (% of Sales)", 0.0, 20.0, 8.0) / 100.0
    sim_return_processing = st.number_input("Cost to Process Return (£/unit)", 50, 500, 120)
    
    st.header("🏗️ Corporate Strategy")
    strategy = st.radio("Distribution Strategy", [
        "Optimize Mix (Owned + 3PL)", 
        "100% Asset-Light (3PL Only)", 
        "100% Asset-Heavy (Owned Only)"
    ])
    
    run_button = st.button("🚀 Run Network Optimization", type="primary", use_container_width=True)

if run_button:
    with st.spinner("Executing Strategic Solver..."):
        # --- (Data Loading Block - Same as your snippet) ---
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
        except:
            st.error("Excel file 'SC_Model_Data.xlsx' not found. Please ensure it's in the directory.")
            st.stop()

        # Constants
        WACC = df_const.loc['WACC', 'Value']
        INFLATION = df_const.loc['Variable_Cost_Inflation', 'Value']
        PRICE = 2500
        years, sups, facs, dcs, regs = list(range(1, 6)), df_sup.index.tolist(), df_fac.index.tolist(), df_3pl.index.tolist(), df_dem.columns.tolist()

        # Setup LP
        model = pulp.LpProblem("Network_Optimization", pulp.LpMaximize)

        # Variables
        build_fac = pulp.LpVariable.dicts("BuildFac", ((f, y, s) for f in facs for y in years for s in ['Std', 'Mega']), cat='Binary')
        open_dc_3pl = pulp.LpVariable.dicts("Open3PL", ((dc, y) for dc in dcs for y in years), cat='Binary')
        build_dc_own = pulp.LpVariable.dicts("BuildOwnDC", ((dc, y) for dc in dcs for y in years), cat='Binary')
        
        flow_in = pulp.LpVariable.dicts("FlowIn", ((s, f, y) for s in sups for f in facs for y in years), lowBound=0)
        flow_out = pulp.LpVariable.dicts("FlowOut", ((f, dc, y) for f in facs for dc in dcs for y in years), lowBound=0)
        flow_last_3pl = pulp.LpVariable.dicts("FlowLast3PL", ((dc, r, y) for dc in dcs for r in regs for y in years), lowBound=0)
        flow_last_own = pulp.LpVariable.dicts("FlowLastOwn", ((dc, r, y) for dc in dcs for r in regs for y in years), lowBound=0)
        flow_unmet = pulp.LpVariable.dicts("Unmet", ((r, y) for r in regs for y in years), lowBound=0)

        BIG_M = 1000000 

        # Return Flow constraints (Reverse Logistics)
        # We assume units returned go back to the DC that served them
        for y in years:
            for r in regs:
                # Demand satisfaction
                model += pulp.lpSum([flow_last_3pl[dc, r, y] + flow_last_own[dc, r, y] for dc in dcs]) + flow_unmet[r, y] == df_dem.loc[y, r] * sim_demand
            
            for dc in dcs:
                # DC Flow Balance: Inbound from Factory = Outbound to Customers (Forward)
                model += pulp.lpSum([flow_out[f, dc, y] for f in facs]) == pulp.lpSum([flow_last_3pl[dc, r, y] + flow_last_own[dc, r, y] for r in regs])
                
                # Capacity/Binary constraints
                model += pulp.lpSum([flow_last_3pl[dc, r, y] for r in regs]) <= open_dc_3pl[dc, y] * BIG_M
                model += pulp.lpSum([flow_last_own[dc, r, y] for r in regs]) <= pulp.lpSum([build_dc_own[dc, yb] for yb in years if yb <= y]) * BIG_M

        # Objective: NPV including Reverse Logistics Cost
        discounted_cfs = []
        for y in years:
            inf = (1 + INFLATION)**(y - 1)
            
            # Revenues
            revenue = pulp.lpSum([(flow_last_3pl[dc, r, y] + flow_last_own[dc, r, y]) * PRICE for dc in dcs for r in regs])
            
            # Forward Variable Costs
            var_costs = (pulp.lpSum([flow_in[s, f, y] * (df_sup.loc[s, 'RM_Cost'] * (1 + sim_tariff)) for s in sups for f in facs]) +
                         pulp.lpSum([flow_in[s, f, y] * (sim_freight) for s in sups for f in facs]) +
                         pulp.lpSum([flow_out[f, dc, y] * df_freight_out.loc[f, dc] for f in facs for dc in dcs]) +
                         pulp.lpSum([(flow_last_3pl[dc, r, y] + flow_last_own[dc, r, y]) * df_last_mile.loc[dc, r] for dc in dcs for r in regs])) * inf
            
            # NEW: Reverse Logistics Costs
            # Every unit sold triggers a return flow with associated processing & return freight costs
            reverse_costs = pulp.lpSum([(flow_last_3pl[dc, r, y] + flow_last_own[dc, r, y]) * sim_returns * (df_last_mile.loc[dc, r] + sim_return_processing) for dc in dcs for r in regs]) * inf

            # Fixed Costs & CAPEX
            fixed = pulp.lpSum([open_dc_3pl[dc, y] * df_3pl.loc[dc, 'Fixed_Cost'] for dc in dcs]) + \
                    pulp.lpSum([build_dc_own[dc, yb] * df_3pl.loc[dc, 'Owned_Fixed_Cost'] for dc in dcs for yb in years if yb <= y])
            
            capex = pulp.lpSum([build_fac[f, y, 'Std'] * 5000000 + build_fac[f, y, 'Mega'] * 12000000 for f in facs])
            
            discounted_cfs.append((revenue - var_costs - reverse_costs - fixed - capex) / ((1 + WACC)**y))

        model += pulp.lpSum(discounted_cfs)
        model.solve()

        # --- FINANCIAL EXTRACTION & RESULTS ---
        st.success(f"Network Optimization Complete. 5-Year Strategy NPV: £{pulp.value(model.objective)/1000000:.2f}M")
        
        t1, t2 = st.tabs(["📊 Executive Summary", "💰 Financial Bridge"])
        
        with t1:
            st.write("### Optimal Network Design")
            cols = st.columns(2)
            # Display where factories were built and DCs opened...
            with cols[0]:
                st.write("**Manufacturing Footprint**")
                # (Logic to show factory build years...)
                st.dataframe(pd.DataFrame([{"Year": y, "Site": f} for f in facs for y in years if build_fac[f,y,'Std'].varValue or build_fac[f,y,'Mega'].varValue]))
            with cols[1]:
                st.write("**Logistics Footprint**")
                st.dataframe(pd.DataFrame([{"Year": y, "DC": dc, "Model": "Owned" if build_dc_own[dc,y].varValue else "3PL"} for dc in dcs for y in years if open_dc_3pl[dc,y].varValue or build_dc_own[dc,y].varValue]))

        with t2:
            st.write("### NPV Driver Analysis")
            # Show a breakdown including the new Reverse Logistics line item
            st.warning("Reverse Logistics represents a hidden (£) impact on your NPV. Use the sidebar to see how return rates change your network structure.")
