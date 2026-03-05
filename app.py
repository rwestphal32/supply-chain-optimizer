import streamlit as st
import pandas as pd
import pulp
import io
import plotly.graph_objects as go

st.set_page_config(page_title="UK Strategic Network: CFO Audit Architect", layout="wide")

# --- 1. DATA FACTORY (RESTORING SPECIFICITY & ASYMMETRY) ---
def generate_rich_template():
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        pd.DataFrame({"Parameter": ["WACC", "Variable_Cost_Inflation", "Tax Rate"], "Value": [0.10, 0.03, 0.25]}).to_excel(writer, sheet_name="Constants", index=False)
        pd.DataFrame({"Supplier": ["Shenzhen_China"], "RM_Cost": [750], "Tariff_Rate": [0.20]}).to_excel(writer, sheet_name="Suppliers", index=False)
        
        # Factories - UNIQUE CAPACITIES & FIXED COSTS
        pd.DataFrame({
            "Site": ["Manchester", "Birmingham", "Glasgow"],
            "Cap_Std": [45000, 60000, 35000],
            "Cap_Mega": [130000, 180000, 110000],
            "Fixed_Cost_Annual": [900000, 1450000, 750000]
        }).to_excel(writer, sheet_name="Facilities", index=False)
        
        # DCs - UNIQUE CAPEX & HANDLING
        pd.DataFrame({
            "DC_Location": ["North_Hub", "Midlands_Hub", "South_Hub"],
            "Fixed_Cost": [480000, 680000, 880000],
            "Variable_Handling_Cost": [45, 50, 65],
            "Owned_Fixed_Cost": [190000, 290000, 390000],
            "Owned_Var_Handling": [20, 22, 28],
            "Owned_CAPEX": [4200000, 6200000, 8200000]
        }).to_excel(writer, sheet_name="3PL_Nodes", index=False)
        
        # Demand (5 Years) - ASYMMETRIC REGIONS
        years, regs = [1,2,3,4,5], ["London", "Scotland", "North_East", "South_West", "NI"]
        demand_matrix = [{"Year": y, "London": 18000*y, "Scotland": 7500*y, "North_East": 9000*y, "South_West": 8500*y, "NI": 3500*y} for y in years]
        pd.DataFrame(demand_matrix).to_excel(writer, sheet_name="Demand", index=False)
        
        # Freight Inbound (China -> UK Factories) - UNIQUE RATES
        pd.DataFrame([{"From": "Shenzhen_China", "Manchester": 195, "Birmingham": 165, "Glasgow": 230}]).to_excel(writer, sheet_name="Freight_Inbound", index=False)
        
        # Freight Outbound (Factory -> DC) - UNIQUE LANES
        pd.DataFrame([
            {"From": "Manchester", "North_Hub": 12, "Midlands_Hub": 48, "South_Hub": 95},
            {"From": "Birmingham", "North_Hub": 48, "Midlands_Hub": 15, "South_Hub": 45},
            {"From": "Glasgow", "North_Hub": 68, "Midlands_Hub": 110, "South_Hub": 185}
        ]).to_excel(writer, sheet_name="Freight_Outbound", index=False)
        
        # Last Mile (DC -> Region) - GEOGRAPHIC ASYMMETRY
        pd.DataFrame([
            {"From": "North_Hub", "London": 130, "Scotland": 75, "North_East": 25, "South_West": 150, "NI": 260},
            {"From": "Midlands_Hub", "London": 70, "Scotland": 150, "North_East": 65, "South_West": 85, "NI": 230},
            {"From": "South_Hub", "London": 25, "Scotland": 280, "North_East": 140, "South_West": 55, "NI": 380}
        ]).to_excel(writer, sheet_name="Last_Mile", index=False)
    return output.getvalue()

# --- 2. UI HEADER ---
st.title("🇬🇧 UK Strategic Network: SPEC-Profit Architect")
with st.expander("📌 Strategic Objective & Optimization Logic", expanded=True):
    st.markdown("""
    **Objective:** Identify the network configuration that maximizes **5-Year NPV**.
    
    **The Solver's Decision Set:**
    * **Profit Max Mode:** The solver treats demand as a ceiling. It will exit a region if the multi-echelon cost-to-serve (Forward + Reverse Logistics) exceeds the unit margin.
    * **Auditability:** Every flow is captured in the **CFO Audit Ledger** with a granular line-item cost breakdown per region.
    """)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("📥 Data Management")
    st.download_button("📥 Download Rich Spec Template", data=generate_rich_template(), file_name="Network_Profit_Template.xlsx")
    uploaded_file = st.file_uploader("Upload PortCo Specs (.xlsx)", type=["xlsx"])
    st.header("🏢 Strategy")
    strategy = st.radio("Asset Policy", ["Optimize Mix", "3PL Only", "Owned Only"])
    service_mode = st.radio("Commercial Policy", ["Capture All Demand", "Profit Max (Rationalize)"])
    st.header("🔄 Reverse Logistics")
    sim_returns = st.slider("Global Returns (%)", 0, 30, 8) / 100.0
    sim_refurb = st.slider("Refurb Cost (£/unit)", 50, 500, 150)
    run_button = st.button("🚀 Solve Strategic Network", type="primary", use_container_width=True)

# --- 4. THE SOLVER ---
if run_button:
    source_data = uploaded_file if uploaded_file else io.BytesIO(generate_rich_template())
    try:
        df_fac = pd.read_excel(source_data, sheet_name='Facilities').set_index('Site')
        df_dem = pd.read_excel(source_data, sheet_name='Demand').set_index('Year')
        df_sup = pd.read_excel(source_data, sheet_name='Suppliers').set_index('Supplier')
        df_3pl = pd.read_excel(source_data, sheet_name='3PL_Nodes').set_index('DC_Location')
        df_f_in = pd.read_excel(source_data, sheet_name='Freight_Inbound').set_index('From')
        df_f_out = pd.read_excel(source_data, sheet_name='Freight_Outbound').set_index('From')
        df_last = pd.read_excel(source_data, sheet_name='Last_Mile').set_index('From')
        df_const = pd.read_excel(source_data, sheet_name='Constants').set_index('Parameter')
    except Exception as e:
        st.error(f"Excel Structure Error: {e}")
        st.stop()

    WACC, PRICE = df_const.loc['WACC', 'Value'], 2500
    years, sups, facs, dcs, regs = [1,2,3,4,5], df_sup.index.tolist(), df_fac.index.tolist(), df_3pl.index.tolist(), df_dem.columns.tolist()

    model = pulp.LpProblem("UK_Strategic_Network", pulp.LpMaximize)

    # Variables
    build_fac = pulp.LpVariable.dicts("BF", ((f, y, s) for f in facs for y in years for s in ['Std', 'Mega']), cat='Binary')
    open_3pl = pulp.LpVariable.dicts("O3", ((d, y) for d in dcs for y in years), cat='Binary')
    build_own = pulp.LpVariable.dicts("BO", ((d, y) for d in dcs for y in years), cat='Binary')
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
            model += pulp.lpSum([f_last[(d, r, y)] for r in regs]) <= (open_3pl[(d, y)] + pulp.lpSum([build_own[(d, yb)] for yb in years if yb <= y])) * 1e7
        for f in facs:
            model += pulp.lpSum([f_in[(s, f, y)] for s in sups]) == pulp.lpSum([f_out[(f, d, y)] for d in dcs])
            cap = pulp.lpSum([build_fac[(f, yb, 'Std')] * df_fac.loc[f, 'Cap_Std'] + build_fac[(f, yb, 'Mega')] * df_fac.loc[f, 'Cap_Mega'] for yb in years if yb <= y])
            model += pulp.lpSum([f_out[(f, d, y)] for d in dcs]) <= cap

    # NPV Calculation
    cashflows = []
    for y in years:
        rev = pulp.lpSum([f_last[(d, r, y)] * PRICE for d in dcs for r in regs])
        c_prod = pulp.lpSum([f_in[(s, f, y)] * (df_sup.loc[s, 'RM_Cost']*1.2 + df_f_in.loc[s, f]) for s in sups for f in facs])
        c_fwd = pulp.lpSum([f_out[(f, d, y)] * df_f_out.loc[f, d] for f in facs for d in dcs]) + \
                pulp.lpSum([f_last[(d, r, y)] * (df_last.loc[d, r] + df_3pl.loc[d, 'Variable_Handling_Cost']) for d in dcs for r in regs])
        c_rev = pulp.lpSum([f_last[(d, r, y)] * sim_returns * (df_last.loc[d, r] + sim_refurb) for d in dcs for r in regs])
        fixed = pulp.lpSum([open_3pl[(d, y)] * df_3pl.loc[d, 'Fixed_Cost'] + build_own[(d, y)] * df_3pl.loc[d, 'Owned_Fixed_Cost'] for d in dcs]) + \
                pulp.lpSum([build_fac[(f, y, sz)] * df_fac.loc[f, 'Fixed_Cost_Annual'] for f in facs for sz in ['Std', 'Mega']])
        capex = pulp.lpSum([build_own[(d, y)] * df_3pl.loc[d, 'Owned_CAPEX'] for d in dcs]) + pulp.lpSum([build_fac[(f, y, sz)] * 5e6 for f in facs for sz in ['Std', 'Mega']])
        cashflows.append((rev - c_prod - c_fwd - c_rev - fixed - capex) / ((1+WACC)**y))

    model += pulp.lpSum(cashflows)
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # --- 5. UI OUTPUTS ---
    st.metric("Total Strategic NPV", f"£{pulp.value(model.objective)/1e6:.2f}M")
    t1, t2, t3, t4 = st.tabs(["🏗️ Build Schedule", "🚚 Volume Flow (Sankey)", "📍 Regional Profitability", "📥 CFO Audit Ledger"])

    with t1:
        st.subheader("Asset Deployment Timeline")
        builds = [{"Year": y, "Asset": a, "Type": "Factory" if a in facs else "Owned DC"} for y in years for a in facs + dcs if (a in facs and any(pulp.value(build_fac[(a, y, sz)]) > 0.5 for sz in ['Std', 'Mega'])) or (a in dcs and pulp.value(build_own[(a, y)]) > 0.5)]
        st.table(pd.DataFrame(builds) if builds else pd.DataFrame([{"Status": "Asset-Light Model Chosen"}]))

    with t2:
        st.subheader("Physical Unit Throughput (5-Year River)")
        nodes = sups + facs + dcs + regs
        n_map = {n: i for i, n in enumerate(nodes)}
        s, t, v = [], [], []
        for sup in sups:
            for fac in facs:
                val = sum([pulp.value(f_in[(sup,fac,y)]) for y in years])
                if val > 1: s.append(n_map[sup]); t.append(n_map[fac]); v.append(val)
        for fac in facs:
            for dc in dcs:
                val = sum([pulp.value(f_out[(fac,dc,y)]) for y in years])
                if val > 1: s.append(n_map[fac]); t.append(n_map[dc]); v.append(val)
        for dc in dcs:
            for reg in regs:
                val = sum([pulp.value(f_last[(dc,reg,y)]) for y in years])
                if val > 1: s.append(n_map[dc]); t.append(n_map[reg]); v.append(val)
        fig = go.Figure(data=[go.Sankey(node=dict(label=nodes, color="blue"), link=dict(source=s, target=t, value=v))])
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        st.subheader("Regional Margin Verification")
        reg_res = []
        for r in regs:
            vol = sum([pulp.value(f_last[(d,r,y)]) for d in dcs for y in years])
            if vol > 0:
                t_rev, t_cost = vol * PRICE, 0
                for y in years:
                    for d in dcs:
                        v = pulp.value(f_last[(d,r,y)])
                        if v > 0:
                            # Forensic lane-specific costing
                            t_cost += v * (df_sup.iloc[0]['RM_Cost']*1.2 + df_last.loc[d,r] + df_3pl.loc[d,'Variable_Handling_Cost'] + sim_returns*(df_last.loc[d,r] + sim_refurb))
                reg_res.append({"Region": r, "Units": int(vol), "Contribution Margin %": round((t_rev-t_cost)/t_rev*100, 1), "Net Profit (£)": t_rev-t_cost})
        st.dataframe(pd.DataFrame(reg_res).sort_values("Contribution Margin %", ascending=False), 
                     column_config={"Contribution Margin %": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100)}, use_container_width=True)

    with t4:
        st.subheader("Transactional Audit Ledger")
        ledger = []
        for y in years:
            for d in dcs:
                for r in regs:
                    v = pulp.value(f_last[(d,r,y)])
                    if v > 0:
                        ledger.append({
                            "Year": y, "Region": r, "Source_DC": d, "Units": v, "Unit_Rev": PRICE,
                            "COGS_Est": df_sup.iloc[0]['RM_Cost']*1.2,
                            "Fwd_Freight": df_last.loc[d,r], "Handling": df_3pl.loc[d,'Variable_Handling_Cost'],
                            "Return_Leakage": sim_returns*(df_last.loc[d,r] + sim_refurb),
                            "Net_Unit_Margin": PRICE - (df_sup.iloc[0]['RM_Cost']*1.2 + df_last.loc[d,r] + df_3pl.loc[d,'Variable_Handling_Cost'] + sim_returns*(df_last.loc[d,r]+sim_refurb))
                        })
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame(ledger).to_excel(writer, sheet_name="Path_Audit", index=False)
            pd.DataFrame(reg_res).to_excel(writer, sheet_name="Regional_Summary", index=False)
        st.download_button("📥 Download Forensic CFO Ledger (.xlsx)", data=output.getvalue(), file_name="Network_Strategic_Audit.xlsx")

else:
    st.info("👈 Set your strategy and click 'Solve'. Download the Spec Template for asymmetric geographic data.")
