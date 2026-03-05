import streamlit as st
import pandas as pd
import pulp
import io
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="UK SC Profitability Architect", layout="wide")

# --- 1. SESSION STATE FOR DATA ---
if 'sc_data' not in st.session_state:
    st.session_state.sc_data = None

# --- 2. HEADER & OBJECTIVE ---
st.title("🇬🇧 UK Strategic Network: Profitability & Flow Architect")
with st.expander("📌 Strategic Mission", expanded=False):
    st.markdown("""
    **Objective:** Identify the network configuration that maximizes **Contribution Margin** and **NPV**.
    **Key Innovation:** This model allows for **Demand Rationalization**. If a region is unprofitable due to high freight or returns, the solver can choose to exit that market.
    """)

# --- 3. SIDEBAR: DATA & LEVERS ---
with st.sidebar:
    st.header("📥 Data Management")
    # Template Generator
    def get_template():
        output = io.BytesIO()
        # Mocking the structure of your SC_Model_Data.xlsx for the demo
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame({"Parameter": ["WACC", "Inflation", "Tax Rate"], "Value": [0.10, 0.03, 0.25]}).to_excel(writer, sheet_name="Constants", index=False)
            pd.DataFrame({"Site": ["Manchester", "Birmingham"], "Cap_Std": [50000, 50000], "Fixed_Cost_Annual": [1e6, 1e6]}).to_excel(writer, sheet_name="Facilities", index=False)
            pd.DataFrame({"Supplier": ["Shenzhen_China"], "RM_Cost": [800], "Tariff_Rate": [0.20]}).to_excel(writer, sheet_name="Suppliers", index=False)
        return output.getvalue()

    st.download_button("📥 Download Template", data=get_template(), file_name="SC_Input_Template.xlsx")
    uploaded_file = st.file_uploader("Upload Network Data", type=["xlsx"])

    st.header("📈 Commercial Policy")
    service_mode = st.radio("Service Strategy", ["Capture All Demand", "Profit Max (Exit Unprofitable Markets)"])
    small_market_returns = st.toggle("Small Markets Pay Own Returns", value=False)
    
    st.header("🔄 Reverse Logistics")
    sim_returns = st.slider("Global Return Rate (%)", 0, 30, 8) / 100.0
    sim_refurb = st.slider("Refurbishment Cost (£/unit)", 50, 500, 150)

    run_button = st.button("🚀 Solve Network Profitability", type="primary", use_container_width=True)

# --- 4. THE SOLVER ENGINE ---
if run_button:
    # (Assuming file exists or use default demo logic for speed)
    file_name = 'SC_Model_Data.xlsx' # Fallback to your local file
    df_fac = pd.read_excel(file_name, sheet_name='Facilities').set_index('Site')
    df_dem = pd.read_excel(file_name, sheet_name='Demand').set_index('Year')
    df_sup = pd.read_excel(file_name, sheet_name='Suppliers').set_index('Supplier')
    df_3pl = pd.read_excel(file_name, sheet_name='3PL_Nodes').set_index('DC_Location')
    df_last_mile = pd.read_excel(file_name, sheet_name='Last_Mile').set_index('From')
    df_freight_out = pd.read_excel(file_name, sheet_name='Freight_Outbound').set_index('From')

    years, sups, facs, dcs, regs = [1,2,3,4,5], df_sup.index.tolist(), df_fac.index.tolist(), df_3pl.index.tolist(), df_dem.columns.tolist()
    PRICE = 2500
    WACC = 0.10

    model = pulp.LpProblem("UK_Profit_Architect", pulp.LpMaximize)

    # Variables
    f_last = pulp.LpVariable.dicts("Flow", (dcs, regs, years), lowBound=0)
    build_fac = pulp.LpVariable.dicts("BuildFac", (facs, years), cat='Binary')
    
    # Decisions & Constraints
    for y in years:
        for r in regs:
            # Mode Switch: Must serve vs Optional
            if service_mode == "Capture All Demand":
                model += pulp.lpSum([f_last[dc, r, y] for dc in dcs]) == df_dem.loc[y, r]
            else:
                model += pulp.lpSum([f_last[dc, r, y] for dc in dcs]) <= df_dem.loc[y, r]

        for dc in dcs:
            # DC Capacity (Simplified for Demo Logic)
            model += pulp.lpSum([f_last[dc, r, y] for r in regs]) <= 100000 

    # Objective: Maximize NPV (Contribution Margin)
    cashflows = []
    for y in years:
        rev = pulp.lpSum([f_last[dc, r, y] * PRICE for dc in dcs for r in regs])
        
        # Forward Freight + RM + DC Handling
        # (Assuming flat £1000 base COGS for demo logic)
        cost_fwd = pulp.lpSum([f_last[dc, r, y] * (1000 + df_last_mile.loc[dc, r] + 50) for dc in dcs for r in regs])
        
        # Reverse Logistics logic
        ret_freight_mult = 0 if small_market_returns else 1
        cost_rev = pulp.lpSum([f_last[dc, r, y] * sim_returns * (df_last_mile.loc[dc, r] * ret_freight_mult + sim_refurb) for dc in dcs for r in regs])
        
        # Fixed Costs
        fixed = pulp.lpSum([build_fac[f, y] * 1000000 for f in facs])
        
        cashflows.append((rev - cost_fwd - cost_rev - fixed) / ((1+WACC)**y))

    model += pulp.lpSum(cashflows)
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # --- 5. VISUALIZATION: THE MONEY MAP ---
    st.header("📊 Executive Results")
    
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Money Flow Map", "📍 Regional Profitability", "🏗️ Infrastructure", "📥 CFO Audit Export"])

    with tab1:
        st.subheader("Where the Money Moves (5-Year Total)")
        # Aggregate flows for Sankey
        t_rev = sum([f_last[dc,r,y].varValue * PRICE for dc in dcs for r in regs for y in years])
        t_fwd = sum([f_last[dc,r,y].varValue * (1000 + df_last_mile.loc[dc,r]) for dc in dcs for r in regs for y in years])
        t_rev_log = sum([f_last[dc,r,y].varValue * sim_returns * (df_last_mile.loc[dc,r] + sim_refurb) for dc in dcs for r in regs for y in years])
        
        fig = go.Figure(data=[go.Sankey(
            node = dict(pad = 15, thickness = 20, line = dict(color = "black", width = 0.5),
              label = ["Gross Revenue", "Operating Margin", "Logistics Costs", "Reverse Logistics", "Net Profit"],
              color = "blue"),
            link = dict(
              source = [0, 0, 2, 1], 
              target = [1, 2, 3, 4],
              value = [t_rev - t_fwd, t_fwd, t_rev_log, t_rev - t_fwd - t_rev_log]
          ))])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Contribution Margin % by Region")
        regional_stats = []
        for r in regs:
            r_vol = sum([f_last[dc,r,y].varValue for dc in dcs for y in years])
            if r_vol > 0:
                r_rev = r_vol * PRICE
                r_cost = sum([f_last[dc,r,y].varValue * (1000 + df_last_mile.loc[dc,r] + (sim_returns * (df_last_mile.loc[dc,r] + sim_refurb))) for dc in dcs for y in years])
                regional_stats.append({"Region": r, "Volume": int(r_vol), "CM %": round((r_rev - r_cost)/r_rev * 100, 1)})
        
        df_reg = pd.DataFrame(regional_stats).sort_values("CM %", ascending=False)
        st.dataframe(df_reg, use_container_width=True)
        st.caption("Low CM% regions are candidates for pricing surcharges or service exit.")

    with tab4:
        st.subheader("Detailed Audit Ledger")
        # Build an audit dataframe of every flow
        audit_data = []
        for y in years:
            for dc in dcs:
                for r in regs:
                    vol = f_last[dc,r,y].varValue
                    if vol > 0:
                        audit_data.append({
                            "Year": y, "From_DC": dc, "To_Region": r, "Units": vol,
                            "Revenue": vol * PRICE, "LastMile_Cost": vol * df_last_mile.loc[dc,r],
                            "Returns_Cost": vol * sim_returns * (df_last_mile.loc[dc,r] + sim_refurb)
                        })
        df_audit = pd.DataFrame(audit_data)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_audit.to_excel(writer, sheet_name="Network_Flow_Audit", index=False)
            df_reg.to_excel(writer, sheet_name="Regional_Profitability", index=False)
        
        st.download_button("📥 Download CFO Audit Ledger (.xlsx)", data=output.getvalue(), file_name="Network_Optimization_Audit.xlsx")

else:
    st.info("👈 Configure your Profit Strategy and click 'Solve' to reveal the financial network.")
