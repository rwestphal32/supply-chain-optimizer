import streamlit as st
import pandas as pd
import pulp
import io

st.set_page_config(page_title="UK Network Architect", layout="wide")

# --- 1. PROBLEM STATEMENT ---
st.title("🇬🇧 UK Strategic Network Design Architect")
with st.container():
    st.markdown("""
    ### Strategic Objective
    **Problem:** fragmented distribution and rising 'Last Mile' costs are eroding the 5-Year NPV. 
    **Task:** Use Mixed-Integer Linear Programming (MILP) to find the optimal balance between **Fixed CAPEX** (building owned sites) and **Variable OPEX** (using 3PLs), while accounting for the circular cost of **Reverse Logistics**.
    """)

# --- 2. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🌍 Global Levers")
    sim_tariff = st.slider("China Tariff Rate (%)", 0.0, 0.5, 0.20)
    sim_freight = st.slider("Ocean Freight (China -> UK)", 50, 500, 150)
    
    st.header("📈 Market Levers")
    sim_demand = st.slider("Demand Growth Multiplier", 0.5, 2.0, 1.0)
    sim_returns = st.slider("Returns Rate (%)", 0, 20, 8) / 100.0
    
    st.header("🏗️ Corporate Strategy")
    strategy = st.radio("Distribution Strategy", [
        "Optimize Mix (Owned + 3PL)", 
        "100% Asset-Light (3PL Only)", 
        "100% Asset-Heavy (Owned Only)"
    ])
    
    run_button = st.button("🚀 Run Strategic Solver", type="primary", use_container_width=True)

if run_button:
    # --- 3. DATA LOADING ---
    file_name = 'SC_Model_Data.xlsx'
    try:
        df_fac = pd.read_excel(file_name, sheet_name='Facilities').set_index('Site')
        df_const = pd.read_excel(file_name, sheet_name='Constants').set_index('Parameter')
        df_dem = pd.read_excel(file_name, sheet_name='Demand').set_index('Year')
        df_sup = pd.read_excel(file_name, sheet_name='Suppliers').set_index('Supplier')
        df_3pl = pd.read_excel(file_name, sheet_name='3PL_Nodes').set_index('DC_Location')
        df_freight_out = pd.read_excel(file_name, sheet_name='Freight_Outbound').set_index('From')
        df_last_mile = pd.read_excel(file_name, sheet_name='Last_Mile').set_index('From')
    except:
        st.error("Missing Data File.")
        st.stop()

    # Constants
    WACC = 0.10
    TAX_RATE = 0.25
    PRICE = 2500
    RET_PRO_COST = 150 # Cost to process one return
    years, facs, dcs, regs = list(range(1, 6)), df_fac.index.tolist(), df_3pl.index.tolist(), df_dem.columns.tolist()

    # --- 4. OPTIMIZATION ENGINE ---
    model = pulp.LpProblem("NPV_Maximizer", pulp.LpMaximize)

    # Decision Vars
    build_dc_own = pulp.LpVariable.dicts("BuildOwn", (dcs, years), cat='Binary')
    use_3pl = pulp.LpVariable.dicts("Use3PL", (dcs, years), cat='Binary')
    flow = pulp.LpVariable.dicts("Flow", (dcs, regs, years), lowBound=0)

    # Constraints
    for y in years:
        for r in regs:
            # Satisfy Demand
            model += pulp.lpSum([flow[dc, r, y] for dc in dcs]) == df_dem.loc[y, r] * sim_demand
        
        for dc in dcs:
            # Strategy Logic
            if strategy == "100% Asset-Light (3PL Only)":
                model += build_dc_own[dc, y] == 0
            elif strategy == "100% Asset-Heavy (Owned Only)":
                model += use_3pl[dc, y] == 0
            
            # Capacity (Simplified for clarity)
            model += pulp.lpSum([flow[dc, r, y] for r in regs]) <= (build_dc_own[dc, y] + use_3pl[dc, y]) * 50000

    # Financial NPV Calculation
    yearly_cashflows = []
    for y in years:
        # Revenue
        rev = pulp.lpSum([flow[dc, r, y] * PRICE for dc in dcs for r in regs])
        
        # Forward Logistics
        fwd_cost = pulp.lpSum([flow[dc, r, y] * df_last_mile.loc[dc, r] for dc in dcs for r in regs])
        
        # Reverse Logistics (Returns Freight + Processing)
        # Returns follow the same path back to the DC
        rev_cost = pulp.lpSum([flow[dc, r, y] * sim_returns * (df_last_mile.loc[dc, r] + RET_PRO_COST) for dc in dcs for r in regs])
        
        # Fixed Costs
        fixed_3pl = pulp.lpSum([use_3pl[dc, y] * 500000 for dc in dcs])
        fixed_own = pulp.lpSum([build_dc_own[dc, y] * 200000 for dc in dcs]) # Owned is cheaper OPEX
        
        # CAPEX
        capex = pulp.lpSum([build_dc_own[dc, y] * 5000000 for dc in dcs])
        
        # Yearly EBITDA-ish
        net_cf = rev - fwd_cost - rev_cost - fixed_3pl - fixed_own - capex
        yearly_cashflows.append(net_cf / ((1 + WACC)**y))

    model += pulp.lpSum(yearly_cashflows)
    model.solve()

    # --- 5. OUTPUT SUBSTANCE ---
    st.metric("Strategy NPV", f"£{pulp.value(model.objective)/1000000:.1f}M")

    tab1, tab2, tab3 = st.tabs(["🚀 Executive Summary", "💰 Financial Bridge", "🔄 Reverse Logistics"])

    with tab1:
        st.subheader("Optimal Network Footprint")
        active_nodes = []
        for y in years:
            for dc in dcs:
                if build_dc_own[dc, y].varValue > 0.5:
                    active_nodes.append({"Year": y, "Location": dc, "Type": "Owned DC (Asset Heavy)"})
                if use_3pl[dc, y].varValue > 0.5:
                    active_nodes.append({"Year": y, "Location": dc, "Type": "3PL Node (Asset Light)"})
        if active_nodes:
            st.table(pd.DataFrame(active_nodes))
        else:
            st.error("No valid network found for these constraints.")

    with tab2:
        st.subheader("5-Year Financial Breakdown")
        # Aggregating data for the bridge
        total_rev = sum([flow[dc, r, y].varValue * PRICE for dc in dcs for r in regs for y in years])
        total_fwd = sum([flow[dc, r, y].varValue * df_last_mile.loc[dc, r] for dc in dcs for r in regs for y in years])
        total_rev_log = sum([flow[dc, r, y].varValue * sim_returns * (df_last_mile.loc[dc, r] + RET_PRO_COST) for dc in dcs for r in regs for y in years])
        total_fixed = sum([(use_3pl[dc, y].varValue * 500000) + (build_dc_own[dc, y].varValue * 200000) for dc in dcs for y in years])
        total_capex = sum([build_dc_own[dc, y].varValue * 5000000 for dc in dcs for y in years])

        df_bridge = pd.DataFrame({
            "Metric": ["Gross Revenue", "Forward Logistics", "Reverse Logistics", "Fixed OPEX", "Total CAPEX", "NET CASH FLOW"],
            "Value (£)": [total_rev, -total_fwd, -total_rev_log, -total_fixed, -total_capex, (total_rev - total_fwd - total_rev_log - total_fixed - total_capex)]
        })
        st.dataframe(df_bridge.style.format({"Value (£)": "£{:,.0f}"}), use_container_width=True)

    with tab3:
        st.subheader("Reverse Logistics Impact")
        st.write(f"Based on a **{sim_returns*100:.1f}%** return rate, the network is processing approximately **{int(sum([df_dem.loc[y,r]*sim_demand for y in years for r in regs]) * sim_returns):,} returned units**.")
        st.info(f"Total cost of failure (Reverse Logistics): £{total_rev_log:,.0f}")
        st.write("This includes return freight and a standard refurbishment cost of £150 per unit.")
else:
    st.info("👈 Set your strategy and return rates in the sidebar, then click 'Run' to see the substance.")
