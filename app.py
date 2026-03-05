import streamlit as st
import pandas as pd
import pulp
import io
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="UK SC Profitability Architect", layout="wide")

# --- 1. PROBLEM STATEMENT ---
st.title("🇬🇧 UK Strategic Network: Profitability & Flow Architect")
with st.expander("📌 Strategic Objective", expanded=False):
    st.markdown("""
    **Objective:** Identify the network configuration that maximizes **Contribution Margin** and **NPV**.
    
    **The "Black Box" Unlocked:**
    * **Rationalization:** In 'Profit Max' mode, the solver will exit regions where the cost of logistics and returns exceeds the margin.
    * **Reverse Logistics:** Models the circular flow of units back to the DC for refurbishment.
    * **Auditability:** Download a full transactional ledger of every move the solver made.
    """)

# --- 2. SIDEBAR: DATA & POLICY ---
with st.sidebar:
    st.header("📥 Data Management")
    
    def get_template():
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame({"Site": ["Manchester", "Birmingham", "London"], "Cap_Std": [50000, 50000, 50000], "Fixed_Cost_Annual": [1e6, 1e6, 1e6]}).to_excel(writer, sheet_name="Facilities", index=False)
            pd.DataFrame({"DC_Location": ["North_Hub", "Midlands_Hub", "South_Hub"], "Fixed_Cost": [5e5, 5e5, 5e5], "Variable_Handling_Cost": [50, 50, 50], "Owned_Fixed_Cost": [2e5, 2e5, 2e5], "Owned_Var_Handling": [25, 25, 25], "Owned_CAPEX": [3e6, 3e6, 3e6]}).to_excel(writer, sheet_name="3PL_Nodes", index=False)
            pd.DataFrame({"Supplier": ["Shenzhen_China"], "RM_Cost": [800], "Tariff_Rate": [0.20], "Inbound_Multiplier": [1.0]}).to_excel(writer, sheet_name="Suppliers", index=False)
            # Demand and freight would follow the structure of your original file
        return output.getvalue()

    st.download_button("📥 Download Input Template", data=get_template(), file_name="Network_Data_Template.xlsx")
    uploaded_file = st.file_uploader("Upload PortCo Data (.xlsx)", type=["xlsx"])

    st.header("📈 Commercial Policy")
    service_mode = st.radio("Service Strategy", ["Capture All Demand", "Profit Max (Rationalize Markets)"])
    
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
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    years, sups, facs, dcs, regs = [1,2,3,4,5], df_sup.index.tolist(), df_fac.index.tolist(), df_3pl.index.tolist(), df_dem.columns.tolist()
    PRICE, WACC = 2500, 0.10

    model = pulp.LpProblem("UK_Profit_Architect", pulp.LpMaximize)

    # Variables
    f_in = pulp.LpVariable.dicts("F_In", (sups, facs, years), lowBound=0)
    f_out = pulp.LpVariable.dicts("F_Out", (facs, dcs, years), lowBound=0)
    f_last = pulp.LpVariable.dicts("F_Last", (dcs, regs, years), lowBound=0)
    build_fac = pulp.LpVariable.dicts("BuildFac", (facs, years), cat='Binary')
    
    # Logic
    for y in years:
        for r in regs:
            total_served = pulp.lpSum([f_last[dc, r, y] for dc in dcs])
            if service_mode == "Capture All Demand":
                model += total_served == df_dem.loc[y, r]
            else:
                model += total_served <= df_dem.loc[y, r]

        for dc in dcs:
            model += pulp.lpSum([f_out[f, dc, y] for f in facs]) == pulp.lpSum([f_last[dc, r, y] for r in regs])
        
        for f in facs:
            model += pulp.lpSum([f_in[s, f, y] for s in sups]) == pulp.lpSum([f_out[f, dc, y] for dc in dcs])
            model += pulp.lpSum([f_out[f, dc, y] for dc in dcs]) <= pulp.lpSum([build_fac[f, yb] for yb in years if yb <= y]) * 100000

    # Financial Components
    cashflows = []
    for y in years:
        rev = pulp.lpSum([f_last[dc, r, y] * PRICE for dc in dcs for r in regs])
        
        # Forward Echelon Costs
        c_rm = pulp.lpSum([f_in[s, f, y] * df_sup.loc[s, 'RM_Cost'] * (1+df_sup.loc[s, 'Tariff_Rate']) for s in sups for f in facs])
        c_log = pulp.lpSum([f_in[s, f, y] * df_freight_in.loc[s, f] for s in sups for f in facs]) + \
                pulp.lpSum([f_out[f, dc, y] * df_freight_out.loc[f, dc] for f in facs for dc in dcs]) + \
                pulp.lpSum([f_last[dc, r, y] * df_last_mile.loc[dc, r] for dc in dcs for r in regs])
        
        # Reverse Logistics
        c_rev = pulp.lpSum([f_last[dc, r, y] * sim_returns * (df_last_mile.loc[dc, r] + sim_refurb) for dc in dcs for r in regs])
        
        # Fixed
        c_fixed = pulp.lpSum([build_fac[f, y] * df_fac.loc[f, 'Fixed_Cost_Annual'] for f in facs])
        
        cashflows.append((rev - c_rm - c_log - c_rev - c_fixed) / ((1+WACC)**y))

    model += pulp.lpSum(cashflows)
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # --- 4. OUTPUT & VISUALS ---
    st.header("📊 Executive Financial Results")
    
    tab1, tab2, tab3 = st.tabs(["💰 Money Flow (Sankey)", "📍 Regional Profitability", "📥 CFO Audit Ledger"])

    with tab1:
        st.subheader("Where the Money Moves (5-Year Total)")
        t_rev = sum([f_last[dc,r,y].varValue * PRICE for dc in dcs for r in regs for y in years])
        t_rm = sum([f_in[s,f,y].varValue * df_sup.loc[s, 'RM_Cost'] * (1+df_sup.loc[s, 'Tariff_Rate']) for s in sups for f in facs for y in years])
        t_log = sum([f_last[dc,r,y].varValue * df_last_mile.loc[dc,r] for dc in dcs for r in regs for y in years]) # Simp.
        t_rev_log = sum([f_last[dc,r,y].varValue * sim_returns * (df_last_mile.loc[dc,r] + sim_refurb) for dc in dcs for r in regs for y in years])
        
        fig = go.Figure(data=[go.Sankey(
            node = dict(label = ["Gross Revenue", "COGS (RM/Tariff)", "Operating Margin", "Logistics", "Reverse Logistics", "Net Profit"], color="blue"),
            link = dict(
                source = [0, 0, 2, 2, 3],
                target = [1, 2, 3, 5, 4],
                value = [t_rm, t_rev - t_rm, t_log, t_rev - t_rm - t_log - t_rev_log, t_rev_log]
            ))])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Contribution Margin (CM) Analysis by Region")
        reg_data = []
        for r in regs:
            vol = sum([f_last[dc,r,y].varValue for dc in dcs for y in years])
            if vol > 0:
                rev = vol * PRICE
                cost = sum([f_last[dc,r,y].varValue * (df_sup.iloc[0]['RM_Cost']*1.2 + df_last_mile.loc[dc,r] + (sim_returns * (df_last_mile.loc[dc,r] + sim_refurb))) for dc in dcs for y in years])
                reg_data.append({"Region": r, "Units Sold": int(vol), "CM %": round((rev-cost)/rev*100, 1), "Net Profit": rev-cost})
        
        df_res = pd.DataFrame(reg_data).sort_values("CM %", ascending=False)
        st.dataframe(df_res.style.background_gradient(subset=['CM %'], cmap='RdYlGn'), use_container_width=True)

    with tab3:
        st.subheader("CFO Data Export")
        # Transactional Log
        log = []
        for y in years:
            for dc in dcs:
                for r in regs:
                    v = f_last[dc,r,y].varValue
                    if v > 0:
                        log.append({"Year": y, "DC": dc, "Region": r, "Units": v, "Rev": v*PRICE, "LastMile": v*df_last_mile.loc[dc,r]})
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame(log).to_excel(writer, sheet_name="Detailed_Flows", index=False)
            df_res.to_excel(writer, sheet_name="Regional_Profitability", index=False)
        st.download_button("📥 Download Transactional Audit (.xlsx)", data=output.getvalue(), file_name="UK_Network_Audit.xlsx")
