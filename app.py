import streamlit as st
import pandas as pd
import pulp
import io
import plotly.graph_objects as go

st.set_page_config(page_title="UK Supply Chain Architect", layout="wide")

# --- 1. PROBLEM STATEMENT / OBJECTIVE ---
st.title("🇬🇧 UK Strategic Network Design Architect")
with st.expander("📌 Strategic Objective & Optimization Logic", expanded=True):
    st.markdown("""
    **Objective:** Maximize the **5-Year Net Present Value (NPV)** of the UK business unit.
    
    **The Optimization Challenge:**
    * **Forward Flow:** Supplier -> Factory (Std/Mega) -> DC (Owned/3PL) -> Region.
    * **Reverse Flow:** Region -> DC (Returns processing and refurbishment).
    * **The Trade-off:** Should we spend **CAPEX** today to build owned infrastructure with lower variable costs, or stay **Asset-Light** using 3PLs with higher variable handling?
    """)

# --- 2. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🌍 Macro Levers")
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
        st.error(f"Data Load Error: {e}")
        st.stop()

    # --- 3. MILP SETUP ---
    WACC = df_const.loc['WACC', 'Value']
    PRICE = 2500
    years = [1, 2, 3, 4, 5]
    sups, facs, dcs, regs = df_sup.index.tolist(), df_fac.index.tolist(), df_3pl.index.tolist(), df_dem.columns.tolist()

    model = pulp.LpProblem("UK_Strategic_Network", pulp.LpMaximize)

    # Binaries for Infrastructure
    build_fac = pulp.LpVariable.dicts("BuildFac", ((f, y, s) for f in facs for y in years for s in ['Std', 'Mega']), cat='Binary')
    open_3pl = pulp.LpVariable.dicts("Open3PL", ((dc, y) for dc in dcs for y in years), cat='Binary')
    build_own_dc = pulp.LpVariable.dicts("BuildOwnDC", ((dc, y) for dc in dcs for y in years), cat='Binary')

    # Continuous Flows (Tuple Keys to avoid KeyError)
    f_in = pulp.LpVariable.dicts("FlowIn", ((s, f, y) for s in sups for f in facs for y in years), lowBound=0)
    f_out = pulp.LpVariable.dicts("FlowOut", ((f, dc, y) for f in facs for dc in dcs for y in years), lowBound=0)
    f_last_3pl = pulp.LpVariable.dicts("Flow3PL", ((dc, r, y) for dc in dcs for r in regs for y in years), lowBound=0)
    f_last_own = pulp.LpVariable.dicts("FlowOwn", ((dc, r, y) for dc in dcs for r in regs for y in years), lowBound=0)

    BIG_M = 1000000

    # Constraints
    for y in years:
        for r in regs:
            # Satisfy Seasonal Demand
            model += pulp.lpSum([f_last_3pl[dc, r, y] + f_last_own[dc, r, y] for dc in dcs]) == df_dem.loc[y, r] * sim_demand
        
        for f in facs:
            # Factory Balance
            model += pulp.lpSum([f_in[s, f, y] for s in sups]) == pulp.lpSum([f_out[f, dc, y] for dc in dcs])
            # Factory Capacity
            cap = pulp.lpSum([build_fac[f, yb, 'Std'] * df_fac.loc[f, 'Cap_Std'] + build_fac[f, yb, 'Mega'] * df_fac.loc[f, 'Cap_Mega'] for yb in years if yb <= y])
            model += pulp.lpSum([f_out[f, dc, y] for dc in dcs]) <= cap

        for dc in dcs:
            # DC Balance
            model += pulp.lpSum([f_out[f, dc, y] for f in facs]) == pulp.lpSum([f_last_3pl[dc, r, y] + f_last_own[dc, r, y] for r in regs])
            
            # Strategy Enforcements
            if strategy == "100% Asset-Light (3PL Only)":
                model += build_own_dc[dc, y] == 0
            elif strategy == "100% Asset-Heavy (Owned Only)":
                model += open_3pl[dc, y] == 0
            
            # Link flow to opened infrastructure
            model += pulp.lpSum([f_last_3pl[dc, r, y] for r in regs]) <= open_3pl[dc, y] * BIG_M
            model += pulp.lpSum([f_last_own[dc, r, y] for r in regs]) <= pulp.lpSum([build_own_dc[dc, yb] for yb in years if yb <= y]) * BIG_M

    # --- 4. NPV CALCULATION (INCLUDING REVERSE LOGISTICS) ---
    cashflows = []
    for y in years:
        # Forward Metrics
        rev = pulp.lpSum([(f_last_3pl[dc, r, y] + f_last_own[dc, r, y]) * PRICE for dc in dcs for r in regs])
        
        # Inbound + Tariff
        cost_in = pulp.lpSum([f_in[s, f, y] * (df_sup.loc[s, 'RM_Cost'] * (1 + sim_tariff) + sim_freight) for s in sups for f in facs])
        
        # Outbound + Last Mile
        cost_out = pulp.lpSum([f_out[f, dc, y] * df_freight_out.loc[f, dc] for f in facs for dc in dcs])
        cost_lm = pulp.lpSum([(f_last_3pl[dc, r, y] + f_last_own[dc, r, y]) * df_last_mile.loc[dc, r] for dc in dcs for r in regs])
        
        # Handling
        cost_hand = pulp.lpSum([f_last_3pl[dc, r, y] * df_3pl.loc[dc, 'Variable_Handling_Cost'] for dc in dcs for r in regs]) + \
                    pulp.lpSum([f_last_own[dc, r, y] * df_3pl.loc[dc, 'Owned_Var_Handling'] for dc in dcs for r in regs])

        # REVERSE LOGISTICS: Cost of failure
        cost_rev = pulp.lpSum([(f_last_3pl[dc, r, y] + f_last_own[dc, r, y]) * sim_returns * (df_last_mile.loc[dc, r] + sim_refurb_cost) for dc in dcs for r in regs])
        
        # Fixed & CAPEX
        fixed = pulp.lpSum([open_3pl[dc, y] * df_3pl.loc[dc, 'Fixed_Cost'] for dc in dcs]) + \
                pulp.lpSum([build_own_dc[dc, yb] * df_3pl.loc[dc, 'Owned_Fixed_Cost'] for dc in dcs for yb in years if yb <= y]) + \
                pulp.lpSum([build_fac[f, yb, sz] * df_fac.loc[f, 'Fixed_Cost_Annual'] for f in facs for yb in years if yb <= y for sz in ['Std', 'Mega']])
        
        capex = pulp.lpSum([build_fac[f, y, 'Std'] * 5000000 + build_fac[f, y, 'Mega'] * 12000000 for f in facs]) + \
                pulp.lpSum([build_own_dc[dc, y] * 3000000 for dc in dcs])

        cashflows.append((rev - cost_in - cost_out - cost_lm - cost_hand - cost_rev - fixed - capex) / ((1 + WACC)**y))

    model += pulp.lpSum(cashflows)
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # --- 5. DASHBOARDING ---
    tab1, tab2, tab3 = st.tabs(["🚀 Executive Summary", "💰 5-Year Financials", "🔄 Reverse Logistics"])

    with tab1:
        st.subheader("NPV Value Bridge")
        
        # Extract total results
        t_rev = sum([pulp.value(f_last_3pl[dc, r, y] + f_last_own[dc, r, y]) * PRICE for dc in dcs for r in regs for y in years])
        t_in = sum([pulp.value(f_in[s, f, y]) * (df_sup.loc[s, 'RM_Cost'] * (1 + sim_tariff) + sim_freight) for s in sups for f in facs for y in years])
        t_out = sum([pulp.value(f_out[f, dc, y]) * df_freight_out.loc[f, dc] for f in facs for dc in dcs for y in years])
        t_lm = sum([pulp.value(f_last_3pl[dc, r, y] + f_last_own[dc, r, y]) * df_last_mile.loc[dc, r] for dc in dcs for r in regs for y in years])
        t_rev_log = sum([pulp.value(f_last_3pl[dc, r, y] + f_last_own[dc, r, y]) * sim_returns * (df_last_mile.loc[dc, r] + sim_refurb_cost) for dc in dcs for r in regs for y in years])
        t_fixed = sum([pulp.value(open_3pl[dc, y] * 500000 + build_own_dc[dc, y] * 200000) for dc in dcs for y in years])
        t_capex = sum([pulp.value(build_fac[f, y, sz] * 5000000) for f in facs for y in years for sz in ['Std', 'Mega']]) # Approx

        # Waterfall
        fig = go.Figure(go.Waterfall(
            orientation = "v",
            measure = ["absolute", "relative", "relative", "relative", "relative", "relative", "total"],
            x = ["Gross Revenue", "Inbound COGS", "Transport & Hand", "Reverse Logistics", "Fixed OPEX", "CAPEX", "Net Cash Flow"],
            text = [f"£{t_rev/1M:.1f}M", f"-£{t_in/1M:.1f}M", f"-£{(t_out+t_lm)/1M:.1f}M", f"-£{t_rev_log/1M:.1f}M", f"-£{t_fixed/1M:.1f}M", f"-£{t_capex/1M:.1f}M", "Total"],
            y = [t_rev, -t_in, -(t_out+t_lm), -t_rev_log, -t_fixed, -t_capex, 0],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig.update_layout(title = "5-Year Cumulative Strategic Waterfall", height=500)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Project NPV", f"£{pulp.value(model.objective)/1000000:.2f}M")
        col2.metric("EBITDA Margin", f"{(t_rev - t_in - t_out - t_lm - t_rev_log - t_fixed)/t_rev*100:.1f}%")
        col3.metric("Return Rate", f"{sim_returns*100:.1f}%")

    with tab2:
        st.subheader("Detailed P&L Table")
        pl_data = {
            "Line Item": ["Gross Revenue", "RM & Tariffs", "Logistics (Inbound/Outbound)", "Reverse Logistics", "Site Fixed Costs", "CAPEX Investment"],
            "Amount (£M)": [t_rev/1e6, -t_in/1e6, -(t_out+t_lm)/1e6, -t_rev_log/1e6, -t_fixed/1e6, -t_capex/1e6]
        }
        st.table(pd.DataFrame(pl_data))

    with tab3:
        st.subheader("The Circular Economy: Return Flow Audit")
        total_returns = int(sum([df_dem.loc[y,r]*sim_demand for y in years for r in regs]) * sim_returns)
        st.write(f"Total units returned over 5 years: **{total_returns:,} units**")
        st.info(f"Financial leakage from returns: £{t_rev_log:,.0f}")

else:
    st.info("👈 Adjust strategy levers and click **Run Strategic Solver** to calculate the 5-Year NPV.")
