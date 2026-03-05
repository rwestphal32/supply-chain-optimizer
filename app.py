import streamlit as st
import pandas as pd
import pulp
import io
import plotly.graph_objects as go

st.set_page_config(page_title="UK Strategic Network SPEC-Profit Architect", layout="wide")

# --- 1. DATA FACTORY (RESTORING HETEROGENEITY) ---
def generate_rich_template():
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Constants
        pd.DataFrame({"Parameter": ["WACC", "Variable_Cost_Inflation", "Tax Rate"], "Value": [0.10, 0.03, 0.25]}).to_excel(writer, sheet_name="Constants", index=False)
        # Suppliers
        pd.DataFrame({"Supplier": ["Shenzhen_China"], "RM_Cost": [750], "Tariff_Rate": [0.20]}).to_excel(writer, sheet_name="Suppliers", index=False)
        # Facilities - UNIQUE CAPACITIES & COSTS
        pd.DataFrame({
            "Site": ["Manchester", "Birmingham", "Glasgow"],
            "Cap_Std": [40000, 55000, 30000],
            "Cap_Mega": [120000, 160000, 100000],
            "Fixed_Cost_Annual": [850000, 1300000, 650000]
        }).to_excel(writer, sheet_name="Facilities", index=False)
        # DCs - UNIQUE CAPEX & HANDLING
        pd.DataFrame({
            "DC_Location": ["North_Hub", "Midlands_Hub", "South_Hub"],
            "Fixed_Cost": [450000, 650000, 850000],
            "Variable_Handling_Cost": [48, 52, 68],
            "Owned_Fixed_Cost": [180000, 280000, 380000],
            "Owned_Var_Handling": [22, 24, 30],
            "Owned_CAPEX": [4500000, 6500000, 8500000]
        }).to_excel(writer, sheet_name="3PL_Nodes", index=False)
        # Demand (5 Years) - HETEROGENEOUS REGIONS
        years, regs = [1,2,3,4,5], ["London", "Scotland", "North_East", "South_West", "NI"]
        demand_matrix = [{"Year": y, "London": 18000*y, "Scotland": 7000*y, "North_East": 9000*y, "South_West": 8500*y, "NI": 3500*y} for y in years]
        pd.DataFrame(demand_matrix).to_excel(writer, sheet_name="Demand", index=False)
        # Freight Inbound (China -> UK Factories)
        pd.DataFrame([{"From": "Shenzhen_China", "Manchester": 185, "Birmingham": 155, "Glasgow": 220}]).to_excel(writer, sheet_name="Freight_Inbound", index=False)
        # Freight Outbound (Factory -> DC)
        pd.DataFrame([
            {"From": "Manchester", "North_Hub": 12, "Midlands_Hub": 48, "South_Hub": 88},
            {"From": "Birmingham", "North_Hub": 48, "Midlands_Hub": 14, "South_Hub": 48},
            {"From": "Glasgow", "North_Hub": 68, "Midlands_Hub": 98, "South_Hub": 155}
        ]).to_excel(writer, sheet_name="Freight_Outbound", index=False)
        # Last Mile (DC -> Region) - UNIQUE LANE COSTS
        pd.DataFrame([
            {"From": "North_Hub", "London": 125, "Scotland": 82, "North_East": 28, "South_West": 145, "NI": 260},
            {"From": "Midlands_Hub", "London": 75, "Scotland": 145, "North_East": 62, "South_West": 82, "NI": 225},
            {"From": "South_Hub", "London": 28, "Scotland": 260, "North_East": 135, "South_West": 52, "NI": 360}
        ]).to_excel(writer, sheet_name="Last_Mile", index=False)
    return output.getvalue()

# --- 2. UI HEADER ---
st.title("🇬🇧 UK Strategic Network: Spec-Profit Architect")
with st.expander("📌 Optimization Objective", expanded=False):
    st.markdown("""
    **Objective:** Identify the network configuration that maximizes **5-Year NPV** using a multi-echelon MILP solver.
    **Volume Sankey:** Traces physical throughput (Units) from Origin to Region.
    **Profit Max Mode:** Allows the solver to exit regions where return-heavy logistics destroy margin.
    """)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("📥 Data Management")
    st.download_button("📥 Download Rich Data Template", data=generate_rich_template(), file_name="Network_Profit_Template.xlsx")
    uploaded_file = st.file_uploader("Upload PortCo Specs (.xlsx)", type=["xlsx"])
    st.header("🏢 Strategy")
    strategy = st.radio("Asset Strategy", ["Optimize Mix", "3PL Only", "Owned Only"])
    service_mode = st.radio("Commercial Policy", ["Capture All Demand", "Profit Max (Rationalize)"])
    st.header("🔄 Reverse Logistics")
    sim_returns = st.slider("Global Returns (%)", 0, 30, 8) / 100.0
    sim_refurb = st.slider("Refurb Cost (£/unit)", 50, 500, 150)
    run_button = st.button("🚀 Solve Strategic Network", type="primary", use_container_width=True)

# --- 4. SOLVER ENGINE ---
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
        st.error(f"Data Schema Error: {e}. Ensure all 8 sheets are properly formatted.")
        st.stop()

    WACC, PRICE = df_const.loc['WACC', 'Value'], 2500
    years, sups, facs, dcs, regs = [1,2,3,4,5], df_sup.index.tolist(), df_fac.index.tolist(), df_3pl.index.tolist(), df_dem.columns.tolist()

    model = pulp.LpProblem("UK_Strategic_Network", pulp.LpMaximize)

    # Variables
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
            model += pulp.lpSum([f_last[(d, r, y)] for r in regs]) <= (open_3pl[(d, y)] + pulp.lpSum([build_own[(d, yb)] for yb in years if yb <= y])) * 1e6
        for f in facs:
            model += pulp.lpSum([f_in[(s, f, y)] for s in sups]) == pulp.lpSum([f_out[(f, d, y)] for d in dcs])
            cap = pulp.lpSum([build_fac[(f, yb, 'Std')] * df_fac.loc[f, 'Cap_Std'] + build_fac[(f, yb, 'Mega')] * df_fac.loc[f, 'Cap_Mega'] for yb in years if yb <= y])
            model += pulp.lpSum([f_out[(f, d, y)] for d in dcs]) <= cap

    # Objective NPV
    cashflows = []
    for y in years:
        rev = pulp.lpSum([f_last[(d, r, y)] * PRICE for d in dcs for r in regs])
        c_prod = pulp.lpSum([f_in[(s, f, y)] * (df_sup.loc[s, 'RM_Cost'] * (1.2) + df_f_in.loc[s, f]) for s in sups for f in facs])
        c_fwd = pulp.lpSum([f_out[(f, d, y)] * df_f_out.loc[f, d] for f in facs for d in dcs]) + \
                pulp.lpSum([f_last[(d, r, y)] * (df_last.loc[d, r] + df_3pl.loc[d, 'Variable_Handling_Cost']) for d in dcs for r in regs])
        c_rev = pulp.lpSum([f_last[(d, r, y)] * sim_returns * (df_last.loc[d, r] + sim_refurb) for d in dcs for r in regs])
        fixed = pulp.lpSum([open_3pl[(d, y)] * df_3pl.loc[d, 'Fixed_Cost'] + build_own[(d, y)] * df_3pl.loc[d, 'Owned_Fixed_Cost'] + pulp.lpSum([build_fac[(f, y, sz)] * df_fac.loc[f, 'Fixed_Cost_Annual'] for sz in ['Std', 'Mega']]) for d in dcs for f in facs])
        capex = pulp.lpSum([build_own[(d, y)] * df_3pl.loc[d, 'Owned_CAPEX'] for d in dcs]) + pulp.lpSum([build_fac[(f, y, 'Std')] * 5e6 + build_fac[(f, y, 'Mega')] * 12e6 for f in facs])
        cashflows.append((rev - c_prod - c_fwd - c_rev - fixed - capex) / ((1 + WACC)**y))

    model += pulp.lpSum(cashflows)
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # --- 5. UI OUTPUTS ---
    st.metric("Total Strategy NPV", f"£{pulp.value(model.objective)/1e6:.2f}M")
    t1, t2, t3, t4 = st.tabs(["🏗️ Build Schedule", "🚚 Volume Flow (Units)", "📍 Regional Profitability", "📥 CFO Audit"])

    with t1:
        builds = []
        for y in years:
            for f in facs:
                for sz in ['Std', 'Mega']:
                    if pulp.value(build_fac[(f, y, sz)]) > 0.5: builds.append({"Year": y, "Asset": f, "Type": f"Factory ({sz})"})
            for d in dcs:
                if pulp.value(build_own[(d, y)]) > 0.5: builds.append({"Year": y, "Asset": d, "Type": "Owned DC"})
        st.table(pd.DataFrame(builds) if builds else pd.DataFrame([{"Build Status": "No infrastructure built (Asset-Light)"}]))

    with t2:
        st.subheader("Physical Volume Throughput (5-Year Units)")
        all_nodes = sups + facs + dcs + regs
        n_map = {n: i for i, n in enumerate(all_nodes)}
        s_idx, t_idx, v_val = [], [], []
        for s in sups:
            for f in facs:
                v = sum([pulp.value(f_in[(s,f,y)]) for y in years])
                if v > 1: s_idx.append(n_map[s]); t_idx.append(n_map[f]); v_val.append(v)
        for f in facs:
            for d in dcs:
                v = sum([pulp.value(f_out[(f,d,y)]) for y in years])
                if v > 1: s_idx.append(n_map[f]); t_idx.append(n_map[d]); v_val.append(v)
        for d in dcs:
            for r in regs:
                v = sum([pulp.value(f_last[(d,r,y)]) for y in years])
                if v > 1: s_idx.append(n_map[d]); t_idx.append(n_map[r]); v_val.append(v)
        fig = go.Figure(data=[go.Sankey(node=dict(label=all_nodes, color="blue"), link=dict(source=s_idx, target=t_idx, value=v_val))])
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        reg_p = []
        for r in regs:
            vol = sum([pulp.value(f_last[(d,r,y)]) for d in dcs for y in years])
            if vol > 0:
                r_rev = vol * PRICE
                r_cost = vol * (df_sup.iloc[0]['RM_Cost']*1.2 + 150 + (sim_returns * (150 + sim_refurb))) # Est for visualization
                reg_p.append({"Region": r, "Units": int(vol), "Contribution Margin %": round((r_rev-r_cost)/r_rev*100, 1)})
        st.dataframe(pd.DataFrame(reg_p).sort_values("Contribution Margin %", ascending=False), 
                     column_config={"Contribution Margin %": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100)}, 
                     use_container_width=True)

    with t4:
        log_ledger = [{"Year": y, "DC": d, "Region": r, "Units": pulp.value(f_last[(d,r,y)]), "LastMile_Cost": pulp.value(f_last[(d,r,y)])*df_last.loc[d,r]} for y in years for d in dcs for r in regs if pulp.value(f_last[(d,r,y)]) > 0]
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame(log_ledger).to_excel(writer, sheet_name="Shipment_Ledger", index=False)
        st.download_button("📥 Download Full CFO Audit (.xlsx)", data=output.getvalue(), file_name="Strategic_Network_Audit.xlsx")
