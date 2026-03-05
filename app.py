import streamlit as st
import pandas as pd
import pulp
import io
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="UK Strategic Profit Architect", layout="wide")

# --- 1. PROBLEM STATEMENT ---
st.title("🇬🇧 UK Strategic Network: Profitability & Flow Architect")
with st.expander("📌 Strategic Objective & Logic", expanded=False):
    st.markdown("""
    **Objective:** Identify the optimal network configuration and commercial footprint to maximize **5-Year NPV**.
    
    **The "Black Box" Unlocked:**
    * **Rationalization:** In 'Profit Max' mode, the solver exits regions where the cost-to-serve (Forward + Reverse Logistics) exceeds the wholesale margin.
    * **Multi-Echelon Optimization:** Flows are optimized across Suppliers → Factories → DCs → Customer Regions.
    * **Reverse Logistics Audit:** Quantifies the cash-flow "leakage" from returns and refurbishment.
    """)

# --- 2. SIDEBAR: DATA & POLICY ---
with st.sidebar:
    st.header("📥 Data Management")
    
    def get_template():
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame({"Parameter": ["WACC", "Inflation", "Tax Rate", "Base_COGS_Fixed"], "Value": [0.10, 0.03, 0.25, 1000]}).to_excel(writer, sheet_name="Constants", index=False)
            pd.DataFrame({"Site": ["Manchester", "Birmingham", "London"], "Cap_Std": [50000, 50000, 50000], "Cap_Mega": [150000, 150000, 150000], "Fixed_Cost_Annual": [1e6, 1e6, 1e6]}).to_excel(writer, sheet_name="Facilities", index=False)
            pd.DataFrame({"DC_Location": ["North_Hub", "Midlands_Hub", "South_Hub"], "Fixed_Cost": [5e5, 5e5, 5e5], "Variable_Handling_Cost": [50, 50, 50], "Owned_Fixed_Cost": [2e5, 2e5, 2e5], "Owned_Var_Handling": [25, 25, 25], "Owned_CAPEX": [3e6, 3e6, 3e6]}).to_excel(writer, sheet_name="3PL_Nodes", index=False)
            pd.DataFrame({"Supplier": ["Shenzhen_China"], "RM_Cost": [800], "Tariff_Rate": [0.20], "Inbound_Multiplier": [1.0]}).to_excel(writer, sheet_name="Suppliers", index=False)
        return output.getvalue()

    st.download_button("📥 Download Input Template", data=get_template(), file_name="Network_Data_Template.xlsx")
    uploaded_file = st.file_uploader("Upload PortCo Data (.xlsx)", type=["xlsx"])

    st.header("📈 Commercial Strategy")
    service_mode = st.radio("Service Strategy", ["Capture All Demand", "Profit Max (Rationalize Markets)"])
    small_market_policy = st.toggle("Small Markets Pay Own Returns", value=False)
    
    st.header("🔄 Reverse Logistics")
    sim_returns = st.slider("Global Return Rate (%)", 0, 30, 8) / 100.0
    sim_refurb = st.slider("Refurbishment Cost (£/unit)", 50, 500, 150)

    run_button = st.button("🚀 Solve Network Profitability", type="primary", use_container_width=True)

# --- 3. THE SOLVER ENGINE ---
if run_button:
    file_name = 'SC_Model_Data.xlsx' if uploaded_file is None else uploaded_file
    try:
        df_fac = pd.read_excel(file_name, sheet_name='Facilities').set_index('Site')
        df_dem = pd.read_excel(file_name, sheet_name='Demand').set_index('Year')
        df_sup = pd.read_excel(file_name, sheet_name='Suppliers').set_index('Supplier')
        df_3pl = pd.read_excel(file_name, sheet_name='3PL_Nodes').set_index('DC_Location')
        df_freight_in = pd.read_excel(file_name, sheet_name='Freight_Inbound').set_index('From')
        df_freight_out = pd.read_excel(file_name, sheet_name='Freight_Outbound').set_index('From')
        df_last_mile = pd.read_excel(file_name, sheet_name='Last_Mile').set_index('From')
        df_const = pd.read_excel(file_name, sheet_name='Constants').set_index('Parameter')
    except Exception as e:
        st.error(f"Excel Structure Error: {e}")
        st.stop()

    # Constants & Sets
    WACC = df_const.loc['WACC', 'Value']
    INFLATION = df_const.loc['Variable_Cost_Inflation', 'Value']
    PRICE = 2500
    years, sups, facs, dcs, regs = [1,2,3,4,5], df_sup.index.tolist(), df_fac.index.tolist(), df_3pl.index.tolist(), df_dem.columns.tolist()

    model = pulp.LpProblem("UK_Strategic_Network", pulp.LpMaximize)

    # 3.1 Decision Variables (Using FLAT dicts for tuple keys to avoid KeyError)
    build_fac = pulp.LpVariable.dicts("BuildFac", ((f, y, s) for f in facs for y in years for s in ['Std', 'Mega']), cat='Binary')
    open_3pl = pulp.LpVariable.dicts("Open3PL", ((dc, y) for dc in dcs for y in years), cat='Binary')
    build_own_dc = pulp.LpVariable.dicts("BuildOwnDC", ((dc, y) for dc in dcs for y in years), cat='Binary')

    f_in = pulp.LpVariable.dicts("F_In", ((s, f, y) for s in sups for f in facs for y in years), lowBound=0)
    f_out = pulp.LpVariable.dicts("F_Out", ((f, dc, y) for f in facs for dc in dcs for y in years), lowBound=0)
    f_last_3pl = pulp.LpVariable.dicts("F_Last_3PL", ((dc, r, y) for dc in dcs for r in regs for y in years), lowBound=0)
    f_last_own = pulp.LpVariable.dicts("F_Last_Own", ((dc, r, y) for dc in dcs for r in regs for y in years), lowBound=0)

    BIG_M = 1000000

    # 3.2 Constraints
    for y in years:
        for r in regs:
            total_served = pulp.lpSum([f_last_3pl[dc, r, y] + f_last_own[dc, r, y] for dc in dcs])
            if service_mode == "Capture All Demand":
                model += total_served == df_dem.loc[y, r]
            else:
                model += total_served <= df_dem.loc[y, r]

        for dc in dcs:
            # DC Balance: Inbound from Factory = Outbound to Regions
            model += pulp.lpSum([f_out[f, dc, y] for f in facs]) == pulp.lpSum([f_last_3pl[dc, r, y] + f_last_own[dc, r, y] for r in regs])
            # Strategy Linking
            model += pulp.lpSum([f_last_3pl[dc, r, y] for r in regs]) <= open_3pl[dc, y] * BIG_M
            model += pulp.lpSum([f_last_own[dc, r, y] for r in regs]) <= pulp.lpSum([build_own_dc[dc, yb] for yb in years if yb <= y]) * BIG_M

        for f in facs:
            # Factory Balance
            model += pulp.lpSum([f_in[s, f, y] for s in sups]) == pulp.lpSum([f_out[f, dc, y] for dc in dcs])
            # Factory Capacity
            cap = pulp.lpSum([build_fac[f, yb, 'Std'] * df_fac.loc[f, 'Cap_Std'] + build_fac[f, yb, 'Mega'] * df_fac.loc[f, 'Cap_Mega'] for yb in years if yb <= y])
            model += pulp.lpSum([f_out[f, dc, y] for dc in dcs]) <= cap

    # 3.3 Objective Function: Discounted Net Cash Flow
    cashflows = []
    for y in years:
        inf = (1 + INFLATION)**(y-1)
        # Revenue
        rev = pulp.lpSum([(f_last_3pl[dc, r, y] + f_last_own[dc, r, y]) * PRICE for dc in dcs for r in regs])
        
        # Variable Echelon Costs
        c_in = pulp.lpSum([f_in[s, f, y] * (df_sup.loc[s, 'RM_Cost'] * (1 + df_sup.loc[s, 'Tariff_Rate']) + df_freight_in.loc[s, f]) for s in sups for f in facs]) * inf
        c_out = pulp.lpSum([f_out[f, dc, y] * df_freight_out.loc[f, dc] for f in facs for dc in dcs]) * inf
        c_lm = pulp.lpSum([(f_last_3pl[dc, r, y] + f_last_own[dc, r, y]) * df_last_mile.loc[dc, r] for dc in dcs for r in regs]) * inf
        c_hand = pulp.lpSum([f_last_3pl[dc, r, y] * df_3pl.loc[dc, 'Variable_Handling_Cost'] + f_last_own[dc, r, y] * df_3pl.loc[dc, 'Owned_Var_Handling'] for dc in dcs for r in regs]) * inf
        
        # Reverse Logistics (Cost of units coming BACK)
        # Apply the Small Market Policy: if true, we exclude return freight for low-volume regions (< 5000 units/year)
        c_rev = 0
        for dc in dcs:
            for r in regs:
                ret_freight = df_last_mile.loc[dc, r]
                if small_market_policy and df_dem.loc[y, r] < 5000:
                    ret_freight = 0 # Customer pays
                c_rev += (f_last_3pl[dc, r, y] + f_last_own[dc, r, y]) * sim_returns * (ret_freight + sim_refurb) * inf

        # Fixed & CAPEX
        fixed = pulp.lpSum([open_3pl[dc, y] * df_3pl.loc[dc, 'Fixed_Cost'] + build_own_dc[dc, y] * df_3pl.loc[dc, 'Owned_Fixed_Cost'] for dc in dcs]) + \
                pulp.lpSum([build_fac[f, y, sz] * df_fac.loc[f, 'Fixed_Cost_Annual'] for f in facs for sz in ['Std', 'Mega']])
        
        capex = pulp.lpSum([build_own_dc[dc, y] * df_3pl.loc[dc, 'Owned_CAPEX'] for dc in dcs]) + \
                pulp.lpSum([build_fac[f, y, 'Std'] * 5e6 + build_fac[f, y, 'Mega'] * 12e6 for f in facs])

        cashflows.append((rev - (c_in + c_out + c_lm + c_hand + c_rev + fixed + capex)) / ((1 + WACC)**y))

    model += pulp.lpSum(cashflows)
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # --- 4. OUTPUTS & VISUALIZATION ---
    st.header("📊 Strategic Financial Performance")
    
    t1, t2, t3, t4 = st.tabs(["💰 Money Map (Sankey)", "📍 Regional Profitability", "🏗️ Infrastructure", "📥 CFO Audit Ledger"])

    with t1:
        st.subheader("Where the Money Moves (5-Year Total)")
        # Extraction
        v_rev = sum([pulp.value(f_last_3pl[dc,r,y] + f_last_own[dc,r,y]) * PRICE for dc in dcs for r in regs for y in years])
        v_rm = sum([pulp.value(f_in[s,f,y]) * (df_sup.loc[s, 'RM_Cost'] * (1 + df_sup.loc[s, 'Tariff_Rate'])) for s in sups for f in facs for y in years])
        v_log = sum([pulp.value(f_out[f,dc,y] * df_freight_out.loc[f,dc] + (f_last_3pl[dc,r,y] + f_last_own[dc,r,y]) * df_last_mile.loc[dc,r]) for f in facs for dc in dcs for r in regs for y in years])
        v_rev_log = sum([pulp.value(f_last_3pl[dc,r,y] + f_last_own[dc,r,y]) * sim_returns * (df_last_mile.loc[dc,r] + sim_refurb) for dc in dcs for r in regs for y in years])
        
        fig = go.Figure(data=[go.Sankey(
            node = dict(label = ["Gross Revenue", "Raw Materials", "Gross Margin", "Logistics Costs", "Reverse Logistics", "Net Operating Profit"], color="blue"),
            link = dict(
                source = [0, 0, 2, 2, 3],
                target = [1, 2, 3, 5, 4],
                value = [v_rm, v_rev - v_rm, v_log, v_rev - v_rm - v_log - v_rev_log, v_rev_log]
            ))])
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.subheader("Granular Contribution Margin (CM) by Region")
        reg_stats = []
        for r in regs:
            vol = sum([pulp.value(f_last_3pl[dc,r,y] + f_last_own[dc,r,y]) for dc in dcs for y in years])
            if vol > 0:
                r_rev = vol * PRICE
                # Approx cost to serve
                r_cogs = vol * (df_sup.iloc[0]['RM_Cost'] * 1.2)
                r_fwd = sum([pulp.value(f_last_3pl[dc,r,y] + f_last_own[dc,r,y]) * df_last_mile.loc[dc,r] for dc in dcs for y in years])
                r_rev_log = vol * sim_returns * (df_last_mile.loc[dc,regs[0]] + sim_refurb) # Approx
                reg_stats.append({"Region": r, "Volume": int(vol), "CM %": round((r_rev - r_cogs - r_fwd - r_rev_log)/r_rev*100, 1), "Total Contribution": r_rev - r_cogs - r_fwd - r_rev_log})
        
        df_reg = pd.DataFrame(reg_stats).sort_values("CM %", ascending=False)
        st.dataframe(df_reg.style.background_gradient(subset=['CM %'], cmap='RdYlGn'), use_container_width=True)
        st.caption("Negative or Low CM% regions represent 'Value Leakage' and should be rationalized.")

    with t4:
        st.subheader("CFO Audit Data Export")
        log_data = []
        for y in years:
            for dc in dcs:
                for r in regs:
                    v = pulp.value(f_last_3pl[dc,r,y] + f_last_own[dc,r,y])
                    if v > 0:
                        log_data.append({"Year": y, "DC": dc, "Region": r, "Units": v, "Rev": v*PRICE, "Logistics_Cost": v*df_last_mile.loc[dc,r]})
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame(log_data).to_excel(writer, sheet_name="Shipment_Ledger", index=False)
            df_reg.to_excel(writer, sheet_name="Regional_Profitability", index=False)
        
        st.download_button("📥 Download Full CFO Audit (.xlsx)", data=output.getvalue(), file_name="UK_Network_Profit_Audit.xlsx")

else:
    st.info("👈 Set your strategy and click 'Solve' to run the high-fidelity profitability model.")
