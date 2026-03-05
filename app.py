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
        pd.DataFrame({"Parameter": ["WACC", "Variable_Cost_Inflation", "Tax Rate"], "Value": [0.10, 0.03, 0.25]}).to_excel(writer, sheet_name="Constants", index=False)
        pd.DataFrame({"Supplier": ["Shenzhen_China"], "RM_Cost": [750], "Tariff_Rate": [0.20]}).to_excel(writer, sheet_name="Suppliers", index=False)
        # Factories - ASYMMETRIC SPECS
        pd.DataFrame({
            "Site": ["Manchester", "Birmingham", "Glasgow"],
            "Cap_Std": [40000, 55000, 30000],
            "Cap_Mega": [120000, 160000, 100000],
            "Fixed_Cost_Annual": [850000, 1350000, 650000]
        }).to_excel(writer, sheet_name="Facilities", index=False)
        # DCs - ASYMMETRIC SPECS
        pd.DataFrame({
            "DC_Location": ["North_Hub", "Midlands_Hub", "South_Hub"],
            "Fixed_Cost": [450000, 650000, 850000],
            "Variable_Handling_Cost": [48, 52, 68],
            "Owned_Fixed_Cost": [180000, 280000, 380000],
            "Owned_Var_Handling": [22, 24, 30],
            "Owned_CAPEX": [4500000, 6500000, 8500000]
        }).to_excel(writer, sheet_name="3PL_Nodes", index=False)
        # Demand (5 Years)
        years, regs = [1,2,3,4,5], ["London", "Scotland", "North_East", "South_West", "NI"]
        demand_matrix = [{"Year": y, "London": 18000*y, "Scotland": 7500*y, "North_East": 9500*y, "South_West": 8000*y, "NI": 3800*y} for y in years]
        pd.DataFrame(demand_matrix).to_excel(writer, sheet_name="Demand", index=False)
        # Freight Echelons
        pd.DataFrame([{"From": "Shenzhen_China", "Manchester": 190, "Birmingham": 160, "Glasgow": 225}]).to_excel(writer, sheet_name="Freight_Inbound", index=False)
        pd.DataFrame([
            {"From": "Manchester", "North_Hub": 12, "Midlands_Hub": 48, "South_Hub": 95},
            {"From": "Birmingham", "North_Hub": 48, "Midlands_Hub": 14, "South_Hub": 48},
            {"From": "Glasgow", "North_Hub": 68, "Midlands_Hub": 110, "South_Hub": 185}
        ]).to_excel(writer, sheet_name="Freight_Outbound", index=False)
        pd.DataFrame([
            {"From": "North_Hub", "London": 130, "Scotland": 72, "North_East": 25, "South_West": 148, "NI": 260},
            {"From": "Midlands_Hub", "London": 70, "Scotland": 148, "North_East": 62, "South_West": 82, "NI": 225},
            {"From": "South_Hub", "London": 25, "Scotland": 280, "North_East": 138, "South_West": 52, "NI": 380}
        ]).to_excel(writer, sheet_name="Last_Mile", index=False)
    return output.getvalue()

# --- 2. HEADER ---
st.title("🇬🇧 UK Strategic Network: Spec-Profit Architect")
with st.expander("📌 Optimization Objective", expanded=True):
    st.markdown("""
    **Objective:** Identify the network configuration that maximizes **5-Year NPV**.
    **Profit Max Mode:** The solver will rationalize (exit) regions where the cost-to-serve exceeds the wholesale margin.
    **CFO Audit:** The ledger provides a lane-specific unit-cost waterfall for verification.
    """)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("📥 Data Management")
    st.download_button("📥 Download Complex Template", data=generate_rich_template(), file_name="Network_Profit_Spec.xlsx")
    uploaded_file = st.file_uploader("Upload Network Specs (.xlsx)", type=["xlsx"])
    st.header("🏢 Asset Policy")
    strategy = st.radio("Asset Strategy", ["Optimize Mix", "3PL Only", "Owned Only"])
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
        st.error(f"Data Schema Error: {e}")
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
        capex = pulp.lpSum([build_own[(d, y)] * df_3pl.loc[d, 'Owned_CAPEX'] for d in dcs]) + pulp.lpSum([build_fac[(f, y, 'Std')] * 5e6 + build_fac[(f, y, 'Mega')] * 12e6 for f in facs])
        cashflows.append((rev - c_prod - c_fwd - c_rev - fixed - capex) / ((1 + WACC)**y))

    model += pulp.lpSum(cashflows)
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # --- 5. UI OUTPUTS ---
    st.metric("Total Strategic NPV", f"£{pulp.value(model.objective)/1e6:.2f}M")
    t1, t2, t3, t4 = st.tabs(["🏗️ Build Schedule", "🚚 Volume Flow (Sankey)", "📍 Regional Profitability", "📥 CFO Audit Ledger"])

    with t1:
        st.subheader("Infrastructure Deployment Plan")
        builds = [{"Year": y, "Asset": a, "Type": "Factory" if a in facs else "Owned DC"} for y in years for a in facs + dcs if (a in facs and any(pulp.value(build_fac[(a, y, sz)]) > 0.5 for sz in ['Std', 'Mega'])) or (a in dcs and pulp.value(build_own[(a, y)]) > 0.5)]
        st.table(pd.DataFrame(builds) if builds else pd.DataFrame([{"Status": "Asset-Light Solution Chosen"}]))

    with t2:
        st.subheader("Physical Unit Journey (5-Year Throughput)")
        
        nodes = sups + facs + dcs + regs
        n_map = {n: i for i, n in enumerate(nodes)}
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
        fig = go.Figure(data=[go.Sankey(node=dict(label=nodes, color="blue"), link=dict(source=s_idx, target=t_idx, value=v_val))])
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        st.subheader("Verified Contribution Margin by Region")
        reg_stats = []
        for r in regs:
            vol = sum([pulp.value(f_last[(d,r,y)]) for d in dcs for y in years])
            if vol > 0:
                t_rev, t_cost = vol * PRICE, 0
                for y in years:
                    for d in dcs:
                        v = pulp.value(f_last[(d,r,y)])
                        if v > 0:
                            # Direct audit of costs for this specific lane
                            t_cost += v * (df_sup.iloc[0]['RM_Cost']*1.2 + df_last.loc[d,r] + df_3pl.loc[d,'Variable_Handling_Cost'] + sim_returns*(df_last.loc[d,r] + sim_refurb))
                reg_stats.append({"Region": r, "Units": int(vol), "Contribution Margin %": round((t_rev-t_cost)/t_rev*100, 1), "Total Margin (£)": t_rev-t_cost})
        df_p = pd.DataFrame(reg_stats).sort_values("Contribution Margin %", ascending=False)
        st.dataframe(df_p, column_config={"Contribution Margin %": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100)}, use_container_width=True)

    with t4:
        st.subheader("Transactional Audit Ledger")
        ledger = []
        for y in years for d in dcs for r in regs:
            v = pulp.value(f_last[(d,r,y)])
            if v > 0:
                ledger.append({
                    "Year": y, "Region": r, "Source_DC": d, "Units": v, "Rev": v*PRICE,
                    "Landed_COGS_Est": v*df_sup.iloc[0]['RM_Cost']*1.2,
                    "Last_Mile_Cost": v*df_last.loc[d,r], "DC_Handling": v*df_3pl.loc[d,'Variable_Handling_Cost'],
                    "Return_Leakage": v*sim_returns*(df_last.loc[d,r] + sim_refurb),
                    "Profit": v*PRICE - (v*(df_sup.iloc[0]['RM_Cost']*1.2 + df_last.loc[d,r] + df_3pl.loc[d,'Variable_Handling_Cost'] + sim_returns*(df_last.loc[d,r]+sim_refurb)))
                })
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame(ledger).to_excel(writer, sheet_name="Audit_Flows", index=False)
        st.download_button("📥 Download CFO Audit Ledger (.xlsx)", data=output.getvalue(), file_name="Network_Strategic_Audit.xlsx")
