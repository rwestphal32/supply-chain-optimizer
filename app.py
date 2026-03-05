import streamlit as st
import pandas as pd
import pulp
import io
import plotly.graph_objects as go

st.set_page_config(page_title="UK Strategic Network Architect", layout="wide")

# --- 1. PROBLEM STATEMENT ---
st.title("🇬🇧 UK Strategic Network: Profitability & Flow Architect")
with st.expander("📌 Strategic Objective & Logic", expanded=False):
    st.markdown("""
    **Objective:** Identify the optimal network configuration to maximize **5-Year NPV**.
    
    **Investor Logic:**
    * **Profit Max Mode:** The solver treats demand as a ceiling. It will only fulfill a unit if the incremental revenue covers the multi-echelon cost-to-serve.
    * **Money Map:** Visualizes the "bleed" from Revenue to Profit via Forward and Reverse Logistics.
    * **Auditability:** Every flow is captured in the CFO Audit Ledger for manual verification.
    """)

# --- 2. SIDEBAR: DATA & POLICY ---
with st.sidebar:
    st.header("📥 Data Management")
    
    def get_template():
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame({"Parameter": ["WACC", "Inflation", "Tax Rate"], "Value": [0.10, 0.03, 0.25]}).to_excel(writer, sheet_name="Constants", index=False)
            pd.DataFrame({"Site": ["Manchester", "Birmingham", "London"], "Cap_Std": [50000, 50000, 50000], "Fixed_Cost_Annual": [1e6, 1e6, 1e6]}).to_excel(writer, sheet_name="Facilities", index=False)
            pd.DataFrame({"DC_Location": ["North_Hub", "Midlands_Hub", "South_Hub"], "Fixed_Cost": [5e5, 5e5, 5e5], "Variable_Handling_Cost": [50, 50, 50]}).to_excel(writer, sheet_name="3PL_Nodes", index=False)
            pd.DataFrame({"Supplier": ["Shenzhen_China"], "RM_Cost": [800], "Tariff_Rate": [0.20]}).to_excel(writer, sheet_name="Suppliers", index=False)
        return output.getvalue()

    st.download_button("📥 Download Data Template", data=get_template(), file_name="Network_Data_Template.xlsx")
    uploaded_file = st.file_uploader("Upload PortCo Data (.xlsx)", type=["xlsx"])

    st.header("📈 Commercial Strategy")
    service_mode = st.radio("Service Strategy", ["Capture All Demand", "Profit Max (Rationalize Markets)"])
    
    st.header("🔄 Reverse Logistics")
    sim_returns = st.slider("Global Return Rate (%)", 0, 30, 8) / 100.0
    sim_refurb = st.slider("Refurbishment Cost (£/unit)", 50, 500, 150)

    run_button = st.button("🚀 Run Network Optimizer", type="primary", use_container_width=True)

# --- 3. THE SOLVER ENGINE ---
if run_button:
    file_name = 'SC_Model_Data.xlsx' if uploaded_file is None else uploaded_file
    try:
        # Loading all 8 standard echelons
        df_fac = pd.read_excel(file_name, sheet_name='Facilities').set_index('Site')
        df_dem = pd.read_excel(file_name, sheet_name='Demand').set_index('Year')
        df_sup = pd.read_excel(file_name, sheet_name='Suppliers').set_index('Supplier')
        df_3pl = pd.read_excel(file_name, sheet_name='3PL_Nodes').set_index('DC_Location')
        df_f_in = pd.read_excel(file_name, sheet_name='Freight_Inbound').set_index('From')
        df_f_out = pd.read_excel(file_name, sheet_name='Freight_Outbound').set_index('From')
        df_last = pd.read_excel(file_name, sheet_name='Last_Mile').set_index('From')
        df_const = pd.read_excel(file_name, sheet_name='Constants').set_index('Parameter')
    except Exception as e:
        st.error(f"Excel Sheet Error: {e}. Ensure all 8 sheets exist.")
        st.stop()

    # Constants
    WACC = df_const.loc['WACC', 'Value']
    PRICE = 2500
    years, sups, facs, dcs, regs = [1,2,3,4,5], df_sup.index.tolist(), df_fac.index.tolist(), df_3pl.index.tolist(), df_dem.columns.tolist()

    model = pulp.LpProblem("UK_Strategic_Optimizer", pulp.LpMaximize)

    # 3.1 Decision Variables
    f_in = pulp.LpVariable.dicts("F_In", ((s, f, y) for s in sups for f in facs for y in years), lowBound=0)
    f_out = pulp.LpVariable.dicts("F_Out", ((f, d, y) for f in facs for d in dcs for y in years), lowBound=0)
    f_last = pulp.LpVariable.dicts("F_Last", ((d, r, y) for d in dcs for r in regs for y in years), lowBound=0)
    build_fac = pulp.LpVariable.dicts("BuildFac", ((f, y) for f in facs for y in years), cat='Binary')

    # 3.2 Constraints
    for y in years:
        for r in regs:
            total_vol = pulp.lpSum([f_last[(d, r, y)] for d in dcs])
            if service_mode == "Capture All Demand":
                model += total_vol == df_dem.loc[y, r]
            else:
                model += total_vol <= df_dem.loc[y, r]

        for d in dcs:
            model += pulp.lpSum([f_out[(f, d, y)] for f in facs]) == pulp.lpSum([f_last[(d, r, y)] for r in regs])
        
        for f in facs:
            model += pulp.lpSum([f_in[(s, f, y)] for s in sups]) == pulp.lpSum([f_out[(f, d, y)] for d in dcs])
            model += pulp.lpSum([f_out[(f, d, y)] for d in dcs]) <= pulp.lpSum([build_fac[(f, yb)] for yb in years if yb <= y]) * 100000

    # 3.3 Objective Function (NPV)
    cashflows = []
    for y in years:
        rev = pulp.lpSum([f_last[(d, r, y)] * PRICE for d in dcs for r in regs])
        
        # Forward Costs (RM + Tariffs + Freight + Handling)
        c_prod = pulp.lpSum([f_in[(s, f, y)] * (df_sup.loc[s, 'RM_Cost'] * (1 + df_sup.loc[s, 'Tariff_Rate']) + df_f_in.loc[s, f]) for s in sups for f in facs])
        c_fwd = pulp.lpSum([f_out[(f, d, y)] * df_f_out.loc[f, d] for f in facs for d in dcs]) + \
                pulp.lpSum([f_last[(d, r, y)] * (df_last.loc[d, r] + df_3pl.loc[d, 'Variable_Handling_Cost']) for d in dcs for r in regs])
        
        # Reverse Logistics (Return Freight + Processing)
        c_rev = pulp.lpSum([f_last[(d, r, y)] * sim_returns * (df_last.loc[d, r] + sim_refurb) for d in dcs for r in regs])
        
        # Infrastructure
        c_fixed = pulp.lpSum([build_fac[(f, y)] * df_fac.loc[f, 'Fixed_Cost_Annual'] for f in facs])
        
        cashflows.append((rev - c_prod - c_fwd - c_rev - c_fixed) / ((1 + WACC)**y))

    model += pulp.lpSum(cashflows)
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # --- 4. OUTPUTS & VISUALIZATION ---
    st.header("📊 Strategic Financial Audit")
    
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Money Flow (Sankey)", "📍 Regional Profitability", "🏗️ Infrastructure", "📥 CFO Audit Ledger"])

    # Data Extraction
    t_rev = sum([pulp.value(f_last[(d,r,y)]) * PRICE for d in dcs for r in regs for y in years])
    t_cogs = sum([pulp.value(f_in[(s,f,y)]) * (df_sup.loc[s, 'RM_Cost'] * (1 + df_sup.loc[s, 'Tariff_Rate'])) for s in sups for f in facs for y in years])
    t_fwd = sum([pulp.value(f_last[(d,r,y)]) * (df_last.loc[d,r] + df_3pl.loc[d,'Variable_Handling_Cost']) for d in dcs for r in regs for y in years])
    t_rev_log = sum([pulp.value(f_last[(d,r,y)]) * sim_returns * (df_last.loc[d,r] + sim_refurb) for d in dcs for r in regs for y in years])
    t_profit = t_rev - t_cogs - t_fwd - t_rev_log

    with tab1:
        st.subheader("Where the Money Flows (5-Year Total)")
        fig = go.Figure(data=[go.Sankey(
            node = dict(label = ["Gross Revenue", "Raw Materials/Tariffs", "Operating Margin", "Logistics (Fwd)", "Reverse Logistics", "Net Profit"], color="blue"),
            link = dict(
                source = [0, 0, 2, 2, 2],
                target = [1, 2, 3, 4, 5],
                value = [t_cogs, t_rev - t_cogs, t_fwd, t_rev_log, t_profit]
            ))])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Profitability by Region")
        reg_stats = []
        for r in regs:
            vol = sum([pulp.value(f_last[(d,r,y)]) for d in dcs for y in years])
            if vol > 0:
                r_rev = vol * PRICE
                # Simple allocation for regional profitability
                r_cost = vol * (df_sup.iloc[0]['RM_Cost']*1.2 + 100 + (sim_returns * (100 + sim_refurb)))
                reg_stats.append({"Region": r, "Units Sold": int(vol), "CM %": round((r_rev-r_cost)/r_rev*100, 1), "Contribution": r_rev-r_cost})
        
        df_reg = pd.DataFrame(reg_stats).sort_values("CM %", ascending=False)
        st.dataframe(df_reg.style.format({"Contribution": "£{:,.0f}"}), use_container_width=True)
        st.caption("Low CM% regions are candidates for service rationalization or surcharges.")

    with tab4:
        st.subheader("CFO Audit Data Export")
        audit_log = []
        for y in years:
            for d in dcs:
                for r in regs:
                    v = pulp.value(f_last[(d,r,y)])
                    if v > 0:
                        audit_log.append({
                            "Year": y, "DC": d, "Region": r, "Units": v, 
                            "Rev": v*PRICE, "Logistics_Cost": v*df_last.loc[d,r],
                            "Return_Leakage": v*sim_returns*(df_last.loc[d,r] + sim_refurb)
                        })
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame(audit_log).to_excel(writer, sheet_name="Transactional_Ledger", index=False)
            df_reg.to_excel(writer, sheet_name="Regional_Profitability", index=False)
        st.download_button("📥 Download Full CFO Audit (.xlsx)", data=output.getvalue(), file_name="Network_Profit_Audit.xlsx")

else:
    st.info("👈 Set your strategy and click 'Run' to reveal the financial network.")
