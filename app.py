import streamlit as st
import pandas as pd
import pulp

st.set_page_config(page_title="UK Supply Chain Optimizer", layout="wide")

st.title("🇬🇧 UK Strategic Network Design Tool")
st.caption("A Value Creation Engine for Multi-Echelon Supply Chain Optimization")

# --- SIDEBAR SCENARIO CONTROLS ---
with st.sidebar:
    st.header("Global Levers")
    sim_tariff = st.slider("China Tariff Rate (%)", 0.0, 0.5, 0.20, 0.05)
    sim_freight = st.slider("Ocean Freight (China -> UK)", 50, 500, 150, 10)
    
    st.header("Market Levers")
    sim_demand = st.slider("Demand Growth Multiplier", 0.5, 2.0, 1.0, 0.1)
    sim_price = st.slider("Unit Selling Price (£)", 1500, 3500, 2500, 100)
    
    st.header("Corporate Strategy")
    strategy = st.radio("Distribution Strategy", [
        "Optimize Mix (Owned + 3PL)", 
        "100% Asset-Light (3PL Only)", 
        "100% Asset-Heavy (Owned Only)"
    ])
    
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
        TAX_RATE = df_const.loc['Tax Rate', 'Value']
        years, sups, facs, dcs, regs = df_dem.index.tolist(), df_sup.index.tolist(), df_fac.index.tolist(), df_3pl.index.tolist(), df_dem.columns.tolist()

        model = pulp.LpProblem("UK_SC_Model", pulp.LpMaximize)
        
        # Binaries
        build_fac = pulp.LpVariable.dicts("BuildFac", ((f, y, s) for f in facs for y in years for s in ['Std', 'Mega']), cat='Binary')
        open_dc_3pl = pulp.LpVariable.dicts("Open3PL", ((dc, y) for dc in dcs for y in years), cat='Binary')
        build_dc_own = pulp.LpVariable.dicts("BuildOwnDC", ((dc, y) for dc in dcs for y in years), cat='Binary')
        
        # Continuous Flows
        flow_in = pulp.LpVariable.dicts("FlowIn", ((s, f, y) for s in sups for f in facs for y in years), lowBound=0)
        flow_out = pulp.LpVariable.dicts("FlowOut", ((f, dc, y) for f in facs for dc in dcs for y in years), lowBound=0)
        flow_last_3pl = pulp.LpVariable.dicts("FlowLast3PL", ((dc, r, y) for dc in dcs for r in regs for y in years), lowBound=0)
        flow_last_own = pulp.LpVariable.dicts("FlowLastOwn", ((dc, r, y) for dc in dcs for r in regs for y in years), lowBound=0)
        flow_unmet = pulp.LpVariable.dicts("Unmet", ((r, y) for r in regs for y in years), lowBound=0)

        BIG_M = 1000000 
        
        # Strategic UI Overrides
        if strategy == "100% Asset-Light (3PL Only)":
            for dc in dcs:
                for y in years:
                    model += build_dc_own[dc, y] == 0
        elif strategy == "100% Asset-Heavy (Owned Only)":
            for dc in dcs:
                for y in years:
                    model += open_dc_3pl[dc, y] == 0

        for y in years:
            for r in regs:
                model += pulp.lpSum([flow_last_3pl[dc, r, y] + flow_last_own[dc, r, y] for dc in dcs]) + flow_unmet[r, y] == df_dem.loc[y, r]
            for f in facs:
                model += pulp.lpSum([flow_in[s, f, y] for s in sups]) == pulp.lpSum([flow_out[f, dc, y] for dc in dcs])
            for dc in dcs:
                # DC Flow Balance
                model += pulp.lpSum([flow_out[f, dc, y] for f in facs]) == pulp.lpSum([flow_last_3pl[dc, r, y] + flow_last_own[dc, r, y] for r in regs])
                
                # Flow locked to capacity/contracts
                model += pulp.lpSum([flow_last_3pl[dc, r, y] for r in regs]) <= open_dc_3pl[dc, y] * BIG_M
                model += pulp.lpSum([flow_last_own[dc, r, y] for r in regs]) <= pulp.lpSum([build_dc_own[dc, yb] for yb in years if yb <= y]) * BIG_M
            
            for f in facs:
                cap = pulp.lpSum([build_fac[f, yb, 'Std'] * df_fac.loc[f, 'Cap_Std'] + build_fac[f, yb, 'Mega'] * df_fac.loc[f, 'Cap_Mega'] for yb in years if yb <= y])
                model += pulp.lpSum([flow_out[f, dc, y] for dc in dcs]) <= cap
        
        for f in facs:
            model += pulp.lpSum([build_fac[f, y, sz] for y in years for sz in ['Std', 'Mega']]) <= 1
        for dc in dcs:
            model += pulp.lpSum([build_dc_own[dc, y] for y in years]) <= 1 # Can only build owned DC once

        discounted_cfs = []
        for y in years:
            inf = (1 + INFLATION)**(y - 1)
            rev = pulp.lpSum([(flow_last_3pl[dc, r, y] + flow_last_own[dc, r, y]) * PRICE for dc in dcs for r in regs])
            var_costs = (pulp.lpSum([flow_in[s, f, y] * (df_sup.loc[s, 'RM_Cost'] * (1 + df_sup.loc[s, 'Tariff_Rate'])) for s in sups for f in facs]) +
                         pulp.lpSum([flow_in[s, f, y] * (df_freight_in.loc[s, f] * df_sup.loc[s, 'Inbound_Multiplier']) for s in sups for f in facs]) +
                         pulp.lpSum([flow_out[f, dc, y] * df_freight_out.loc[f, dc] for f in facs for dc in dcs]) +
                         pulp.lpSum([(flow_last_3pl[dc, r, y] + flow_last_own[dc, r, y]) * df_last_mile.loc[dc, r] for dc in dcs for r in regs]) +
                         pulp.lpSum([flow_last_3pl[dc, r, y] * df_3pl.loc[dc, 'Variable_Handling_Cost'] for dc in dcs for r in regs]) +
                         pulp.lpSum([flow_last_own[dc, r, y] * df_3pl.loc[dc, 'Owned_Var_Handling'] for dc in dcs for r in regs])) * inf
            
            unmet = pulp.lpSum([flow_unmet[r, y] * 5000 for r in regs])
            
            fixed = pulp.lpSum([open_dc_3pl[dc, y] * df_3pl.loc[dc, 'Fixed_Cost'] for dc in dcs]) + \
                    pulp.lpSum([build_dc_own[dc, yb] * df_3pl.loc[dc, 'Owned_Fixed_Cost'] for dc in dcs for yb in years if yb <= y]) + \
                    pulp.lpSum([build_fac[f, yb, sz] * df_fac.loc[f, 'Fixed_Cost_Annual'] for f in facs for yb in years if yb <= y for sz in ['Std', 'Mega']])
            
            capex = pulp.lpSum([build_fac[f, y, 'Std'] * CAPEX_STD + build_fac[f, y, 'Mega'] * CAPEX_MEGA for f in facs]) + \
                    pulp.lpSum([build_dc_own[dc, y] * df_3pl.loc[dc, 'Owned_CAPEX'] for dc in dcs])
                    
            discounted_cfs.append((rev - var_costs - unmet - fixed - capex) / ((1 + WACC)**y))

        model += pulp.lpSum(discounted_cfs)
        model.solve()

        # --- FINANCIAL EXTRACTION ---
        t_rev, t_mat, t_fin, t_fout, t_lm, t_hand, t_ffac, t_fdc, t_capex = 0,0,0,0,0,0,0,0,0
        for y in years:
            inf = (1 + INFLATION)**(y - 1)
            t_rev += sum((flow_last_3pl[dc, r, y].varValue + flow_last_own[dc, r, y].varValue) * PRICE for dc in dcs for r in regs)
            t_mat += sum(flow_in[s, f, y].varValue * (df_sup.loc[s, 'RM_Cost'] * (1 + df_sup.loc[s, 'Tariff_Rate'])) for s in sups for f in facs) * inf
            t_fin += sum(flow_in[s, f, y].varValue * (df_freight_in.loc[s, f] * df_sup.loc[s, 'Inbound_Multiplier']) for s in sups for f in facs) * inf
            t_fout += sum(flow_out[f, dc, y].varValue * df_freight_out.loc[f, dc] for f in facs for dc in dcs) * inf
            t_lm += sum((flow_last_3pl[dc, r, y].varValue + flow_last_own[dc, r, y].varValue) * df_last_mile.loc[dc, r] for dc in dcs for r in regs) * inf
            
            t_hand += sum(flow_last_3pl[dc, r, y].varValue * df_3pl.loc[dc, 'Variable_Handling_Cost'] for dc in dcs for r in regs) * inf
            t_hand += sum(flow_last_own[dc, r, y].varValue * df_3pl.loc[dc, 'Owned_Var_Handling'] for dc in dcs for r in regs) * inf
            
            t_ffac += sum(build_fac[f, yb, sz].varValue * df_fac.loc[f, 'Fixed_Cost_Annual'] for f in facs for yb in years if yb <= y for sz in ['Std', 'Mega'])
            t_fdc += sum(open_dc_3pl[dc, y].varValue * df_3pl.loc[dc, 'Fixed_Cost'] for dc in dcs)
            t_fdc += sum(build_dc_own[dc, yb].varValue * df_3pl.loc[dc, 'Owned_Fixed_Cost'] for dc in dcs for yb in years if yb <= y)
            
            t_capex += sum(build_fac[f, y, 'Std'].varValue * CAPEX_STD + build_fac[f, y, 'Mega'].varValue * CAPEX_MEGA for f in facs)
            t_capex += sum(build_dc_own[dc, y].varValue * df_3pl.loc[dc, 'Owned_CAPEX'] for dc in dcs)

        # Advanced Financials
        gm = t_rev - (t_mat + t_fin + t_fout + t_lm + t_hand)
        corp_opex = t_rev * 0.22
        ebitda = gm - (t_ffac + t_fdc + corp_opex)
        depreciation = t_capex
        ebit = ebitda - depreciation
        taxes = ebit * TAX_RATE if ebit > 0 else 0
        nopat = ebit - taxes
        roic = (nopat / t_capex * 100) if t_capex > 0 else 0

        # --- UI TABS ---
        tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "💰 Detailed P&L", "🚚 Network Flow"])

        with tab1:
            st.subheader("Key Performance Indicators")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Project NPV", f"£{pulp.value(model.objective)/1000000:.1f}M")
            c2.metric("Gross Margin", f"{gm/t_rev*100:.1f}%" if t_rev > 0 else "0.0%")
            c3.metric("True EBITDA", f"{ebitda/t_rev*100:.1f}%" if t_rev > 0 else "0.0%")
            c4.metric("ROIC (Project ROI)", f"{roic:.1f}%" if t_capex > 0 else "N/A")

            colA, colB = st.columns(2)
            with colA:
                st.write("### Factory Build Schedule")
                fac_log = [{"Year": y, "Site": f, "Type": sz} for f in facs for y in years for sz in ['Std', 'Mega'] if build_fac[f, y, sz].varValue == 1.0]
                if fac_log: st.table(pd.DataFrame(fac_log))
                else: st.warning("No factories built.")
            
            with colB:
                st.write("### Owned DC Build Schedule")
                dc_log = [{"Year": y, "Distribution Center": dc} for dc in dcs for y in years if build_dc_own[dc, y].varValue == 1.0]
                if dc_log: st.table(pd.DataFrame(dc_log))
                else: st.info("Asset-Light Strategy Maintained (No DCs Built).")

        with tab2:
            st.subheader("5-Year Cumulative Income Statement")
            values = [t_rev, -t_mat, -t_fin, -t_fout, -t_hand, -t_lm, gm, -t_ffac, -t_fdc, -corp_opex, ebitda, -depreciation, ebit, -taxes, nopat, -t_capex]
            pct_rev = [(v / t_rev * 100) if t_rev > 0 else 0 for v in values]
