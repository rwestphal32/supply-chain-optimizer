import streamlit as st
import pandas as pd
import pulp

st.set_page_config(page_title="Supply Chain Network Optimizer", layout="wide")

st.title("🌍 Global Supply Chain Network Optimizer")
st.markdown("Adjust the macroeconomic variables on the left, then run the optimization to see how the physical network reacts.")

# --- SIDEBAR INTERFACE ---
st.sidebar.header("Scenario Planning Levers")
sim_tariff = st.sidebar.slider("China Import Tariff (%)", min_value=0.0, max_value=0.5, value=0.20, step=0.05)
sim_freight = st.sidebar.slider("Ocean Freight: China -> UK (£)", min_value=40, max_value=300, value=150, step=10)
sim_demand_multiplier = st.sidebar.slider("Demand Volume Multiplier", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

if st.sidebar.button("🚀 Run Optimization", type="primary"):
    
    with st.spinner("Running MILP Optimization Engine..."):
        # 1. Load Data
        file_name = 'SC_Model_Data.xlsx'
        df_fac = pd.read_excel(file_name, sheet_name='Facilities').set_index('Site')
        df_const = pd.read_excel(file_name, sheet_name='Constants').set_index('Parameter')
        df_dem = pd.read_excel(file_name, sheet_name='Demand').set_index('Year')
        df_sup = pd.read_excel(file_name, sheet_name='Suppliers').set_index('Supplier')
        df_3pl = pd.read_excel(file_name, sheet_name='3PL_Nodes').set_index('DC_Location')
        df_freight_in = pd.read_excel(file_name, sheet_name='Freight_Inbound').set_index('From')
        df_freight_out = pd.read_excel(file_name, sheet_name='Freight_Outbound').set_index('From')
        df_last_mile = pd.read_excel(file_name, sheet_name='Last_Mile').set_index('From')

        # 2. APPLY THE SLIDER OVERRIDES TO THE DATA
        df_sup.at['Shenzhen_China', 'Tariff_Rate'] = sim_tariff
        for f in df_freight_in.columns:
            df_freight_in.at['Shenzhen_China', f] = sim_freight
        df_dem = df_dem * sim_demand_multiplier

        # Constants
        PRICE = df_const.loc['Selling Price', 'Value']
        WACC = df_const.loc['WACC', 'Value']
        INFLATION = df_const.loc['Variable_Cost_Inflation', 'Value']
        CAPEX_STD = df_const.loc['CAPEX_Std', 'Value']
        CAPEX_MEGA = df_const.loc['CAPEX_Mega', 'Value']

        # Sets
        years = df_dem.index.tolist()
        sups = df_sup.index.tolist()
        facs = df_fac.index.tolist()
        dcs = df_3pl.index.tolist()
        regs = df_dem.columns.tolist()

        # Initialize Model
        model = pulp.LpProblem("UK_ScaleUp_3_Echelon", pulp.LpMaximize)

        # Variables
        build_fac = pulp.LpVariable.dicts("Build", ((f, y, s) for f in facs for y in years for s in ['Std', 'Mega']), cat='Binary')
        open_dc = pulp.LpVariable.dicts("OpenDC", ((dc, y) for dc in dcs for y in years), cat='Binary')
        flow_in = pulp.LpVariable.dicts("FlowIn", ((s, f, y) for s in sups for f in facs for y in years), lowBound=0, cat='Continuous')
        flow_out = pulp.LpVariable.dicts("FlowOut", ((f, dc, y) for f in facs for dc in dcs for y in years), lowBound=0, cat='Continuous')
        flow_last = pulp.LpVariable.dicts("FlowLast", ((dc, r, y) for dc in dcs for r in regs for y in years), lowBound=0, cat='Continuous')
        flow_unmet = pulp.LpVariable.dicts("Unmet", ((r, y) for r in regs for y in years), lowBound=0, cat='Continuous')

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

        # Objective Function
        discounted_cfs = []
        tot_revenue, tot_bom_tariff, tot_freight_in, tot_freight_out = 0, 0, 0, 0
        tot_last_mile, tot_3pl_handling, tot_fixed_fac, tot_fixed_3pl, tot_capex = 0, 0, 0, 0, 0

        for y in years:
            inf_factor = (1 + INFLATION)**(y - 1)
            
            revenue = pulp.lpSum([flow_last[dc, r, y] * PRICE for dc in dcs for r in regs])
            bom_y = pulp.lpSum([flow_in[s, f, y] * (df_sup.loc[s, 'RM_Cost'] * (1 + df_sup.loc[s, 'Tariff_Rate'])) for s in sups for f in facs])
            fin_y = pulp.lpSum([flow_in[s, f, y] * (df_freight_in.loc[s, f] * df_sup.loc[s, 'Inbound_Multiplier']) for s in sups for f in facs])
            fout_y = pulp.lpSum([flow_out[f, dc, y] * df_freight_out.loc[f, dc] for f in facs for dc in dcs])
            lm_y = pulp.lpSum([flow_last[dc, r, y] * df_last_mile.loc[dc, r] for dc in dcs for r in regs])
            unmet_pen = pulp.lpSum([flow_unmet[r, y] * df_last_mile.loc['Unmet Penalty', r] for r in regs])
            hand_y = pulp.lpSum([flow_last[dc, r, y] * df_3pl.loc[dc, 'Variable_Handling_Cost'] for dc in dcs for r in regs])
            
            f3pl_y = pulp.lpSum([open_dc[dc, y] * df_3pl.loc[dc, 'Fixed_Cost'] for dc in dcs])
            ffac_y = pulp.lpSum([build_fac[f, yb, sz] * df_fac.loc[f, 'Fixed_Cost_Annual'] for f in facs for yb in years if yb <= y for sz in ['Std', 'Mega']])
            capex_y = pulp.lpSum([build_fac[f, y, 'Std'] * CAPEX_STD + build_fac[f, y, 'Mega'] * CAPEX_MEGA for f in facs])
            
            total_var = (bom_y + fin_y + fout_y + lm_y + hand_y) * inf_factor
            yearly_cf = revenue - total_var - unmet_pen - f3pl_y - ffac_y - capex_y
            discounted_cfs.append(yearly_cf / ((1 + WACC)**y))

        model += pulp.lpSum(discounted_cfs)
        model.solve()

        # --- EXTRACT ACTUAL P&L NUMBERS ---
        for y in years:
            inf_factor = (1 + INFLATION)**(y - 1)
            tot_revenue += sum(flow_last[dc, r, y].varValue * PRICE for dc in dcs for r in regs)
            tot_bom_tariff += sum(flow_in[s, f, y].varValue * (df_sup.loc[s, 'RM_Cost'] * (1 + df_sup.loc[s, 'Tariff_Rate'])) for s in sups for f in facs) * inf_factor
            tot_freight_in += sum(flow_in[s, f, y].varValue * (df_freight_in.loc[s, f] * df_sup.loc[s, 'Inbound_Multiplier']) for s in sups for f in facs) * inf_factor
            tot_freight_out += sum(flow_out[f, dc, y].varValue * df_freight_out.loc[f, dc] for f in facs for dc in dcs) * inf_factor
            tot_last_mile += sum(flow_last[dc, r, y].varValue * df_last_mile.loc[dc, r] for dc in dcs for r in regs) * inf_factor
            tot_3pl_handling += sum(flow_last[dc, r, y].varValue * df_3pl.loc[dc, 'Variable_Handling_Cost'] for dc in dcs for r in regs) * inf_factor
            tot_fixed_fac += sum(build_fac[f, yb, sz].varValue * df_fac.loc[f, 'Fixed_Cost_Annual'] for f in facs for yb in years if yb <= y for sz in ['Std', 'Mega'])
            tot_fixed_3pl += sum(open_dc[dc, y].varValue * df_3pl.loc[dc, 'Fixed_Cost'] for dc in dcs)
            tot_capex += sum(build_fac[f, y, 'Std'].varValue * CAPEX_STD + build_fac[f, y, 'Mega'].varValue * CAPEX_MEGA for f in facs)

        tot_variable_costs = tot_bom_tariff + tot_freight_in + tot_freight_out + tot_last_mile + tot_3pl_handling
        gross_margin = tot_revenue - tot_variable_costs
        gm_pct = (gross_margin / tot_revenue) * 100 if tot_revenue > 0 else 0
        supply_chain_ebitda = gross_margin - (tot_fixed_fac + tot_fixed_3pl)
        corporate_opex = tot_revenue * 0.22
        true_ebitda = supply_chain_ebitda - corporate_opex
        ebitda_pct = (true_ebitda / tot_revenue) * 100 if tot_revenue > 0 else 0

        # --- STREAMLIT DASHBOARD OUTPUT ---
        st.success(f"✅ Optimization Complete! Final NPV: £{pulp.value(model.objective):,.0f}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total 5-Year Revenue", f"£{tot_revenue/1000000:,.1f}M")
        col2.metric("Gross Margin", f"£{gross_margin/1000000:,.1f}M", f"{gm_pct:.1f}%")
        col3.metric("True EBITDA", f"£{true_ebitda/1000000:,.1f}M", f"{ebitda_pct:.1f}%")

        st.subheader("Global Sourcing Strategy (Year 5 Full Run-Rate)")
        y = 5 
        flow_data = []
        for s in sups:
            for f in facs:
                vol_in = flow_in[s, f, y].varValue
                if vol_in and vol_in > 0:
                    flow_data.append({"Source": s, "Factory": f, "Units Shipped": int(vol_in)})
        
        if flow_data:
            st.dataframe(pd.DataFrame(flow_data), use_container_width=True)
        else:
            st.warning("No production in Year 5 (Unit Economics may be negative).")
