import streamlit as st
import pandas as pd
import pulp

st.set_page_config(page_title="UK Supply Chain Optimizer", layout="wide")

st.title("🇬🇧 UK Strategic Network Design Tool")
st.caption("A Value Creation Engine for Multi-Echelon Supply Chain Optimization")

# --- SIDEBAR SCENARIO CONTROLS ---
# Note: If you don't see this, click the '>' arrow in the top left!
with st.sidebar:
    st.header("Global Levers")
    sim_tariff = st.slider("China Tariff Rate (%)", 0.0, 0.5, 0.20, 0.05)
    sim_freight = st.slider("Ocean Freight (China -> UK)", 50, 500, 150, 10)
    
    st.header("Market Levers")
    sim_demand = st.slider("Demand Growth Multiplier", 0.5, 2.0, 1.0, 0.1)
    sim_price = st.slider("Unit Selling Price (£)", 1500, 3500, 2500, 100)
    
    # This is the button you need to click!
    run_button = st.button("🚀 Re-Optimize Network", type="primary", use_container_width=True)

if run_button:
    with st.spinner("Solving Mixed-Integer Linear Program..."):
        # 1. LOAD DATA
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
            st.error(f"Error loading Excel file: {e}")
            st.stop()

        # 2. OVERRIDE WITH SLIDERS
        df_sup.at['Shenzhen_China', 'Tariff_Rate'] = sim_tariff
        for f in df_freight_in.columns:
            df_freight_in.at['Shenzhen_China', f] = sim_freight
        df_dem = df_dem * sim_demand
        PRICE = sim_price

        # 3. SETUP OPTIMIZATION
        WACC = df_const.loc['WACC', 'Value']
        INFLATION = df_const.loc['Variable_Cost_Inflation', 'Value']
        CAPEX_STD = df_const.loc['CAPEX_Std', 'Value']
        CAPEX_MEGA = df_const.loc['CAPEX_Mega', 'Value']
        years, sups, facs, dcs, regs = df_dem.index.tolist(), df_sup.index.tolist(), df_fac.index.tolist(), df_3pl.index.tolist(), df_dem.columns.tolist()

        model = pulp.LpProblem("UK_SC_Model", pulp.LpMaximize)
        build_fac = pulp.LpVariable.dicts("Build", ((f, y, s) for f in facs for y in years for s in ['Std', 'Mega']), cat='Binary')
        open_dc = pulp.LpVariable.dicts("OpenDC", ((dc, y) for dc in dcs for y in years), cat='Binary')
        flow_in = pulp.LpVariable.dicts("FlowIn", ((s, f, y) for s in sups for f in facs for y in years), lowBound=0)
        flow_out = pulp.LpVariable.dicts("FlowOut", ((f, dc, y) for f in facs for dc in dcs for y in years), lowBound=0)
        flow_last = pulp.LpVariable.dicts("FlowLast", ((dc, r, y) for dc in dcs for r in regs for y in years), lowBound=0)
        flow_unmet = pulp.LpVariable.dicts("Unmet", ((r, y) for r in regs for y in years), lowBound=0)

        # Constraints
        BIG_M = 1000000 
        for y in years:
            for r in regs:
                model += pulp.lpSum([flow_last[dc, r, y] for dc in dcs]) + flow_unmet[r, y] == df_dem.loc[y, r]
            for f in facs:
                model += pulp.lpSum([flow_in[s, f, y] for s in sups]) == pulp.lpSum([flow_out[f, dc, y] for dc in dcs])
            for dc in dcs:
                model += pulp.lpSum([flow_out[f, dc, y] for f in facs]) == pulp.lpSum([flow_last[dc, r, y] for r in regs])
                model += pulp.lpSum([flow_last[dc, r, y] for r in regs]) <= open_dc[dc, y] * BIG_M
            for f in facs:
                cap = pulp.lpSum([build_fac[f, yb, 'Std'] * df_fac.loc[f, 'Cap_Std'] + build_fac[f, yb, 'Mega'] * df_fac.loc[f, 'Cap_Mega'] for yb in years if yb <= y])
                model += pulp.lpSum([flow_out[f, dc, y] for dc in dcs]) <= cap
        for f in facs:
            model += pulp.lpSum([build_fac[f, y, sz] for y in years for sz in ['Std', 'Mega']]) <= 1

        discounted_cfs = []
        for y in years:
            inf = (1 + INFLATION)**(y - 1)
            rev = pulp.lpSum([flow_last[dc, r, y] * PRICE for dc in dcs for r in regs])
            var_costs = (pulp.lpSum([flow_in[s, f, y] * (df_sup.loc[s, 'RM_Cost'] * (1 + df_sup.loc[s, 'Tariff_Rate'])) for s in sups for f in facs]) +
                         pulp.lpSum([flow_in[s, f, y] * (df_freight_in.loc[s, f] * df_sup.loc[s, 'Inbound_Multiplier']) for s in sups for f in facs]) +
                         pulp.lpSum([flow_out[f, dc, y] * df_freight_out.loc[f, dc] for f in facs for dc in dcs]) +
                         pulp.lpSum([flow_last[dc, r, y] * (df_last_mile.loc[dc, r] + df_3pl.loc[dc, 'Variable_Handling_Cost']) for dc in dcs for r in regs])) * inf
            unmet = pulp.lpSum([flow_unmet[r, y] * 5000 for r in regs])
            fixed = pulp.lpSum([open_dc[dc, y] * df_3pl.loc[dc, 'Fixed_Cost'] for dc in dcs]) + pulp.lpSum([build_fac[f, yb, sz] * df_fac.loc[f, 'Fixed_Cost_Annual'] for f in facs for yb in years if yb <= y for sz in ['Std', 'Mega']])
            capex = pulp.lpSum([build_fac[f, y, 'Std'] * CAPEX_STD + build_fac[f, y, 'Mega'] * CAPEX_MEGA for f in facs])
            discounted_cfs.append((rev - var_costs - unmet - fixed - capex) / ((1 + WACC)**y))

        model += pulp.lpSum(discounted_cfs)
        model.solve()

        # --- FINANCIAL EXTRACTION ---
        t_rev, t_mat, t_fin, t_fout, t_lm, t_3pl, t_ffac, t_f3pl, t_capex = 0,0,0,0,0,0,0,0,0
        for y in years:
            inf = (1 + INFLATION)**(y - 1)
            t_rev += sum(flow_last[dc, r, y].varValue * PRICE for dc in dcs for r in regs)
            t_mat += sum(flow_in[s, f, y].varValue * (df_sup.loc[s, 'RM_Cost'] * (1 + df_sup.loc[s, 'Tariff_Rate'])) for s in sups for f in facs) * inf
            t_fin += sum(flow_in[s, f, y].varValue * (df_freight_in.loc[s, f] * df_sup.loc[s, 'Inbound_Multiplier']) for s in sups for f in facs) * inf
            t_fout += sum(flow_out[f, dc, y].varValue * df_freight_out.loc[f, dc] for f in facs for dc in dcs) * inf
            t_lm += sum(flow_last[dc, r, y].varValue * df_last_mile.loc[dc, r] for dc in dcs for r in regs) * inf
            t_3pl += sum(flow_last[dc, r, y].varValue * df_3pl.loc[dc, 'Variable_Handling_Cost'] for dc in dcs for r in regs) * inf
            t_ffac += sum(build_fac[f, yb, sz].varValue * df_fac.loc[f, 'Fixed_Cost_Annual'] for f in facs for yb in years if yb <= y for sz in ['Std', 'Mega'])
            t_f3pl += sum(open_dc[dc, y].varValue * df_3pl.loc[dc, 'Fixed_Cost'] for dc in dcs)
            t_capex += sum(build_fac[f, y, 'Std'].varValue * CAPEX_STD + build_fac[f, y, 'Mega'].varValue * CAPEX_MEGA for f in facs)

        # --- UI TABS ---
        tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "💰 Detailed P&L", "🚚 Network Flow"])

        with tab1:
            st.subheader("Key Performance Indicators")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Project NPV", f"£{pulp.value(model.objective)/1000000:.1f}M")
            c2.metric("Total Revenue", f"£{t_rev/1000000:.1f}M")
            gm = t_rev - (t_mat + t_fin + t_fout + t_lm + t_3pl)
            c3.metric("Gross Margin", f"{gm/t_rev*100:.1f}%")
            ebitda = gm - (t_ffac + t_f3pl + (t_rev * 0.22))
            c4.metric("True EBITDA", f"{ebitda/t_rev*100:.1f}%")

            st.write("### Recommended Build Schedule")
            build_log = []
            for f in facs:
                for y in years:
                    for sz in ['Std', 'Mega']:
                        if build_fac[f, y, sz].varValue == 1.0:
                            build_log.append({"Year": y, "Site": f, "Type": sz, "Capacity": df_fac.loc[f, f'Cap_{sz}']})
            if build_log: st.table(pd.DataFrame(build_log))
            else: st.warning("No facility construction recommended for this scenario.")

        with tab2:
            st.subheader("5-Year Cumulative Income Statement")
            pl_data = {
                "Item": ["Revenue", "Raw Materials & Tariffs", "Inbound Freight", "Outbound Freight", "3PL Handling", "Last Mile Delivery", "GROSS MARGIN", "Facility Fixed Costs", "3PL Fixed Costs", "Corporate OpEx (22%)", "TRUE EBITDA", "CAPEX"],
                "Value (£)": [t_rev, -t_mat, -t_fin, -t_fout, -t_3pl, -t_lm, gm, -t_ffac, -t_f3pl, -(t_rev*0.22), ebitda, -t_capex]
            }
            df_pl = pd.DataFrame(pl_data)
            st.dataframe(df_pl, use_container_width=True, hide_index=True)

        with tab3:
            st.subheader("Year 5 Operational Flow Map")
            flow_data = []
            for s in sups:
                for f in facs:
                    if flow_in[s, f, 5].varValue > 0:
                        for dc in dcs:
                            if flow_out[f, dc, 5].varValue > 0:
                                for r in regs:
                                    if flow_last[dc, r, 5].varValue > 0:
                                        flow_data.append({"Supplier": s, "Factory": f, "3PL DC": dc, "Region": r, "Units": int(flow_last[dc, r, 5].varValue)})
            st.dataframe(pd.DataFrame(flow_data), use_container_width=True)
else:
    st.info("👈 Open the sidebar on the left and click 'Re-Optimize' to start.")
