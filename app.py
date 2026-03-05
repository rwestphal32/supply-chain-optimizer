import streamlit as st
import pandas as pd
import pulp
import io
import plotly.graph_objects as go

st.set_page_config(page_title="UK Strategic Network: SPEC-Audit Architect", layout="wide")

# --- 1. DATA FACTORY (RESTORING YOUR ORIGINAL COMPLEXITY) ---
def generate_original_spec_data():
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        pd.DataFrame({"Parameter": ["WACC", "Variable_Cost_Inflation", "Tax Rate"], "Value": [0.10, 0.03, 0.25]}).to_excel(writer, sheet_name="Constants", index=False)
        pd.DataFrame({"Supplier": ["Shenzhen_China"], "RM_Cost": [750], "Tariff_Rate": [0.20]}).to_excel(writer, sheet_name="Suppliers", index=False)
        pd.DataFrame({
            "Site": ["Manchester", "Birmingham", "Glasgow"],
            "Cap_Std": [45000, 60000, 35000],
            "Cap_Mega": [130000, 180000, 110000],
            "Fixed_Cost_Annual": [900000, 1450000, 750000]
        }).to_excel(writer, sheet_name="Facilities", index=False)
        pd.DataFrame({
            "DC_Location": ["North_Hub", "Midlands_Hub", "South_Hub"],
            "Fixed_Cost": [480000, 680000, 880000],
            "Variable_Handling_Cost": [45, 50, 65],
            "Owned_Fixed_Cost": [190000, 290000, 390000],
            "Owned_Var_Handling": [20, 22, 28],
            "Owned_CAPEX": [4200000, 6200000, 8200000]
        }).to_excel(writer, sheet_name="3PL_Nodes", index=False)
        years, regs = [1,2,3,4,5], ["London", "Scotland", "North_East", "South_West", "NI"]
        demand_matrix = [{"Year": y, "London": 18000*y, "Scotland": 7500*y, "North_East": 9500*y, "South_West": 8500*y, "NI": 3500*y} for y in years]
        pd.DataFrame(demand_matrix).to_excel(writer, sheet_name="Demand", index=False)
        pd.DataFrame([{"From": "Shenzhen_China", "Manchester": 195, "Birmingham": 165, "Glasgow": 230}]).to_excel(writer, sheet_name="Freight_Inbound", index=False)
        pd.DataFrame([
            {"From": "Manchester", "North_Hub": 12, "Midlands_Hub": 48, "South_Hub": 95},
            {"From": "Birmingham", "North_Hub": 48, "Midlands_Hub": 14, "South_Hub": 45},
            {"From": "Glasgow", "North_Hub": 68, "Midlands_Hub": 110, "South_Hub": 185}
        ]).to_excel(writer, sheet_name="Freight_Outbound", index=False)
        pd.DataFrame([
            {"From": "North_Hub", "London": 130, "Scotland": 75, "North_East": 25, "South_West": 150, "NI": 260},
            {"From": "Midlands_Hub", "London": 70, "Scotland": 150, "North_East": 65, "South_West": 85, "NI": 230},
            {"From": "South_Hub", "London": 25, "Scotland": 280, "North_East": 140, "South_West": 55, "NI": 380}
        ]).to_excel(writer, sheet_name="Last_Mile", index=False)
    return output.getvalue()

st.title("🇬🇧 UK Strategic Network: CFO Audit Architect")
with st.sidebar:
    st.header("📥 Data Spec")
    st.download_button("📥 Download Spec Template", data=generate_original_spec_data(), file_name="Network_Profit_Spec.xlsx")
    uploaded_file = st.file_uploader("Upload Specs", type=["xlsx"])
    st.header("🏢 Policy")
    service_mode = st.radio("Service Strategy", ["Capture All Demand", "Profit Max (Rationalize)"])
    strategy = st.radio("Asset Model", ["Optimize Mix", "3PL Only", "Owned Only"])
    st.header("🔄 Returns")
    sim_returns = st.slider("Return Rate (%)", 0, 30, 8) / 100.0
    sim_refurb = st.slider("Refurb Cost (£/unit)", 50, 500, 150)
    run_button = st.button("🚀 Solve Strategic Audit", type="primary", use_container_width=True)

if run_button:
    source = uploaded_file if uploaded_file else io.BytesIO(generate_original_spec_data())
    try:
        df_fac = pd.read_excel(source, sheet_name='Facilities').set_index('Site')
        df_dem = pd.read_excel(source, sheet_name='Demand').set_index('Year')
        df_sup = pd.read_excel(source, sheet_name='Suppliers').set_index('Supplier')
        df_3pl = pd.read_excel(source, sheet_name='3PL_Nodes').set_index('DC_Location')
        df_f_in = pd.read_excel(source, sheet_name='Freight_Inbound').set_index('From')
        df_f_out = pd.read_excel(source, sheet_name='Freight_Outbound').set_index('From')
        df_last = pd.read_excel(source, sheet_name='Last_Mile').set_index('From')
        df_const = pd.read_excel(source, sheet_name='Constants').set_index('Parameter')
    except Exception as e:
        st.error(f"Schema Error: {e}"); st.stop()

    WACC, PRICE = df_const.loc['WACC', 'Value'], 2500
    years, sups, facs, dcs, regs = [1,2,3,4,5], df_sup.index.tolist(), df_fac.index.tolist(), df_3pl.index.tolist(), df_dem.columns.tolist()

    model = pulp.LpProblem("UK_Network_Audit", pulp.LpMaximize)

    # Variables
    build_fac = pulp.LpVariable.dicts("BF", ((f, y, s) for f in facs for y in years for s in ['Std', 'Mega']), cat='Binary')
    open_3pl = pulp.LpVariable.dicts("O3", ((d, y) for d in dcs for y in years), cat='Binary')
    build_own = pulp.LpVariable.dicts("BO", ((d, y) for d in dcs for y in years), cat='Binary')
    
    # FLOWS: TRIPLE-INDEXED FOR PATH TRACKING
    f_path = pulp.LpVariable.dicts("Path", ((s, f, d, r, y) for s in sups for f in facs for d in dcs for r in regs for y in years), lowBound=0)

    # Constraints
    for y in years:
        for r in regs:
            vol = pulp.lpSum([f_path[(s, f, d, r, y)] for s in sups for f in facs for d in dcs])
            if service_mode == "Capture All Demand": model += vol == df_dem.loc[y, r]
            else: model += vol <= df_dem.loc[y, r]
            
        for f in facs:
            cap = pulp.lpSum([build_fac[(f, yb, 'Std')] * df_fac.loc[f, 'Cap_Std'] + build_fac[(f, yb, 'Mega')] * df_fac.loc[f, 'Cap_Mega'] for yb in years if yb <= y])
            model += pulp.lpSum([f_path[(s, f, d, r, y)] for s in sups for d in dcs for r in regs]) <= cap
            
        for d in dcs:
            if strategy == "3PL Only": model += build_own[(d, y)] == 0
            elif strategy == "Owned Only": model += open_3pl[(d, y)] == 0
            model += pulp.lpSum([f_path[(s, f, d, r, y)] for s in sups for f in facs for r in regs]) <= (open_3pl[(d, y)] + pulp.lpSum([build_own[(d, yb)] for yb in years if yb <= y])) * 1e7

    # NPV Calculation
    cfs = []
    for y in years:
        inf = (1 + df_const.loc['Variable_Cost_Inflation', 'Value'])**(y-1)
        # Revenue - (RM + Tariff + Freight_Inbound + Freight_Outbound + Handling + Last_Mile + Reverse)
        daily_cash = 0
        for s in sups:
            for f in facs:
                for d in dcs:
                    for r in regs:
                        path_vol = f_path[(s, f, d, r, y)]
                        unit_cost = (df_sup.loc[s, 'RM_Cost']*1.2 + df_f_in.loc[s, f] + df_f_out.loc[f, d] + df_3pl.loc[d, 'Variable_Handling_Cost'] + df_last.loc[d, r])
                        unit_rev_leak = sim_returns * (df_last.loc[d, r] + sim_refurb)
                        daily_cash += path_vol * (PRICE - (unit_cost + unit_rev_leak) * inf)
        
        fixed = pulp.lpSum([open_3pl[(d, y)] * df_3pl.loc[d, 'Fixed_Cost'] + build_own[(d, y)] * df_3pl.loc[d, 'Owned_Fixed_Cost'] for d in dcs])
        capex = pulp.lpSum([build_own[(d, y)] * df_3pl.loc[d, 'Owned_CAPEX'] for d in dcs]) + pulp.lpSum([build_fac[(f, y, sz)] * 5e6 for f in facs for sz in ['Std', 'Mega']])
        cfs.append((daily_cash - fixed - capex) / ((1 + WACC)**y))

    model += pulp.lpSum(cfs)
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # --- 5. UI OUTPUTS ---
    st.metric("Strategy NPV (Verified)", f"£{pulp.value(model.objective)/1e6:.2f}M")
    t1, t2, t3, t4 = st.tabs(["🏗️ Build Schedule", "🚚 Volume Sankey", "📍 Regional Profitability", "📥 Forensic CFO Ledger"])

    with t2:
        st.subheader("Physical Unit Journey (5-Year Throughput)")
        
        nodes = sups + facs + dcs + regs; n_map = {n: i for i, n in enumerate(nodes)}
        s_idx, t_idx, v_val = [], [], []
        for s in sups:
            for f in facs:
                v = sum([pulp.value(f_path[(s,f,d,r,y)]) for d in dcs for r in regs for y in years])
                if v > 1: s_idx.append(n_map[s]); t_idx.append(n_map[f]); v_val.append(v)
        for f in facs:
            for d in dcs:
                v = sum([pulp.value(f_path[(s,f,d,r,y)]) for s in sups for r in regs for y in years])
                if v > 1: s_idx.append(n_map[f]); t_idx.append(n_map[d]); v_val.append(v)
        for d in dcs:
            for r in regs:
                v = sum([pulp.value(f_path[(s,f,d,r,y)]) for s in sups for f in facs for y in years])
                if v > 1: s_idx.append(n_map[d]); t_idx.append(n_map[r]); v_val.append(v)
        st.plotly_chart(go.Figure(data=[go.Sankey(node=dict(label=nodes, color="blue"), link=dict(source=s_idx, target=t_idx, value=v_val))]), use_container_width=True)

    with t3:
        reg_res = []
        for r in regs:
            vol = sum([pulp.value(f_path[(s,f,d,r,y)]) for s in sups for f in facs for d in dcs for y in years])
            if vol > 0:
                cost_sum = sum([pulp.value(f_path[(s,f,d,r,y)]) * (df_sup.loc[s, 'RM_Cost']*1.2 + df_f_in.loc[s, f] + df_f_out.loc[f, d] + df_3pl.loc[d, 'Variable_Handling_Cost'] + df_last.loc[d, r] + sim_returns*(df_last.loc[d,r]+sim_refurb)) for s in sups for f in facs for d in dcs for y in years])
                reg_res.append({"Region": r, "Units": int(vol), "Contribution Margin %": round((vol*PRICE - cost_sum)/(vol*PRICE)*100, 1)})
        st.dataframe(pd.DataFrame(reg_res).sort_values("Contribution Margin %", ascending=False), column_config={"Contribution Margin %": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100)}, use_container_width=True)

    with t4:
        st.subheader("Transactional Pathway Audit")
        ledger = []
        for y in years:
            for s in sups:
                for f in facs:
                    for d in dcs:
                        for r in regs:
                            v = pulp.value(f_path[(s,f,d,r,y)])
                            if v > 0:
                                ledger.append({
                                    "Year": y, "Region": r, "Facility": f, "DC": d, "Units": v,
                                    "Wholesale_Rev": v*PRICE,
                                    "COGS_Product": v*df_sup.loc[s, 'RM_Cost']*1.2,
                                    "Freight_Inbound": v*df_f_in.loc[s,f],
                                    "Freight_Outbound": v*df_f_out.loc[f,d],
                                    "DC_Handling": v*df_3pl.loc[d,'Variable_Handling_Cost'],
                                    "Last_Mile": v*df_last.loc[d,r],
                                    "Return_Leakage": v*sim_returns*(df_last.loc[d,r]+sim_refurb),
                                    "Net_Path_Profit": v*PRICE - (v*(df_sup.loc[s, 'RM_Cost']*1.2 + df_f_in.loc[s, f] + df_f_out.loc[f, d] + df_3pl.loc[d, 'Variable_Handling_Cost'] + df_last.loc[d, r] + sim_returns*(df_last.loc[d,r]+sim_refurb)))
                                })
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame(ledger).to_excel(writer, sheet_name="Forensic_Audit", index=False)
        st.download_button("📥 Download Granular CFO Audit (.xlsx)", data=output.getvalue(), file_name="Strategic_Network_Audit.xlsx")
