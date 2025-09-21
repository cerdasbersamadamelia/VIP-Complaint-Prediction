# ================= app.py =================
# import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime

# ---------- PAGE SETUP ----------
st.set_page_config(layout='wide', page_title="VIP Complaint Prediction Dashboard")
st.markdown("<style>.block-container{padding-top:1rem;padding-bottom:1rem;padding-left:2rem;padding-right:2rem;max-width:100%}</style>", unsafe_allow_html=True)

# import the function from data_processing.py
from data_processing import get_prediction_results

# ========== CUSTOM CSS ==========
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 100% !important;
        }
        .plotly-chart {
            height: 300px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Auto-refresh every 5 minutes (300 seconds)
st.markdown("""<meta http-equiv="refresh" content="300">""", unsafe_allow_html=True)

# ========== MAIN PAGE ==========
results_df = get_prediction_results()

# ========== SIDEBAR ==========
with st.sidebar:
    # st.image("logo.png", width=100)
    st.title("VIP Complaint Prediction Dashboard")

    # --- SITE FILTER ---
    site_options = ['All'] + sorted(results_df['site_id'].unique().tolist())
    selected_site = st.selectbox('Filter by Site', site_options)
    filtered_df = results_df if selected_site == 'All' else results_df[results_df['site_id'] == selected_site]

    # --- ROOT CAUSE FILTER ---
    if 'root_cause' in filtered_df.columns:
        root_options = ['All'] + sorted(filtered_df['root_cause'].dropna().unique().tolist())
        selected_root = st.selectbox('Filter by Root Cause', root_options)
        if selected_root != 'All':
            filtered_df = filtered_df[filtered_df['root_cause'] == selected_root]

    # --- CATEGORY FILTER ---
    if "pred_category" in filtered_df.columns:
        category_options = ["All"] + sorted([str(x) for x in filtered_df['pred_category'].dropna().unique()])
        selected_cat = st.selectbox("Filter by Category", category_options)
        if selected_cat != "All":
            filtered_df = filtered_df[filtered_df['pred_category'] == selected_cat]

    # Description
    with st.expander("📢 Project Info"):
        st.markdown("""
    - **Goal:** Predict VIP complaints 24h ahead & identify root causes  
    - **Inputs (6 CSV):**
        - KPI Timeseries
        - Alarms
        - Topology
        - VIP Tickets
        - Events
        - Weather
    - **Outputs:** 
        - Level 1: Complaint Prediction
        - Level 2: Complaint Category
        - Level 3: Root Cause Recommendation
    - **Benefit:** Proactive handling → improve SLA & customer experience
    """)

# ========== ROW 1 ==========
row1 = st.columns(3)

# Site Location Map
with row1[0]:
    st.subheader('Site Location Map')
    if {'latitude', 'longitude', 'site_id', 'pred_24h'}.issubset(filtered_df.columns) and not filtered_df.empty:
        site_summary = filtered_df.groupby(['site_id', 'latitude', 'longitude', 'pred_24h']).size().reset_index(name='complaint_count')
        color_map = {'Yes': 'red', 'No': 'green'}
        fig_map = px.scatter_mapbox(
            site_summary,
            lat='latitude', lon='longitude',
            size='complaint_count',
            color='pred_24h',
            color_discrete_map=color_map,
            hover_name='site_id',
            hover_data={'complaint_count': True, 'latitude': False, 'longitude': False},
            zoom=10, mapbox_style='open-street-map'
        )
        fig_map.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5), height=200, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": True})
    else:
        st.info("No data.")

# Sub Root Cause Hierarchy
with row1[1]:
    st.subheader("Sub Root Cause Hierarchy")
    if {'root_cause', 'sub_root_cause'}.issubset(filtered_df.columns) and not filtered_df.empty:
        sub_rc_temp = filtered_df['sub_root_cause'].str.replace(r"\(.*\)","",regex=True).str.strip()
        hierarchy_df = filtered_df.assign(sub_root_cause_clean=sub_rc_temp).groupby(['root_cause','sub_root_cause_clean']).size().reset_index(name='Count')
        if hierarchy_df.empty:
            st.info("No data.")
        else:
            fig_treemap = px.treemap(
                hierarchy_df,
                path=['root_cause','sub_root_cause_clean'],
                values='Count',
                color='root_cause',
                color_discrete_sequence=px.colors.sequential.Cividis
            )
            fig_treemap.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5), height=200, margin=dict(t=0,b=0,l=0,r=0))
            st.plotly_chart(fig_treemap, use_container_width=True)
    else:
        st.info("No data.")

# Complaint Trend per Hour
with row1[2]:
    st.subheader('Complaint Trend (per hour)')
    if 'timestamp' in filtered_df.columns and not filtered_df.empty:
        trend = filtered_df.copy()
        trend['hour'] = trend['timestamp'].dt.floor('H')
        hourly = trend.groupby('hour')['pred_24h'].apply(lambda x: (x=='Yes').sum()).reset_index()
        hourly.columns = ['hour','complaints']

        if hourly['complaints'].sum() == 0:  # no actual complaints
            st.info("No data.")
        else:
            fig_trend = px.line(hourly, x='hour', y='complaints', markers=True)
            fig_trend.update_layout(margin=dict(t=0,b=0,l=0,r=0), height=200)
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No data.")

# ========== ROW 2 ==========
row2 = st.columns(3)

# Root Cause Breakdown
with row2[0]:
    st.subheader('Root Cause Breakdown')
    if 'root_cause' in filtered_df.columns and not filtered_df.empty:
        rc_counts = filtered_df['root_cause'].value_counts().reset_index()
        rc_counts.columns = ['Root Cause', 'Count']
        rc_counts = rc_counts.sort_values(by='Count', ascending=False)
        if rc_counts.empty:
            st.info("No data.")
        else:
            fig_bar = px.bar(rc_counts, x='Count', y='Root Cause', text='Count', color='Root Cause',
                             color_discrete_sequence=px.colors.diverging.PiYG, orientation='h')
            fig_bar.update_layout(yaxis=dict(title=''), xaxis=dict(title='Count'),
                                  legend=dict(orientation="h", yanchor="bottom", y=-0.6, xanchor="center", x=0.3),
                                  margin=dict(t=0,b=0,l=0,r=0), height=200)
            fig_bar.update_xaxes(title_text="", showticklabels=False)
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No data.")

# Distribution of Complaint Categories
with row2[1]:
    st.subheader('Complaint Categories')
    if 'pred_category' in filtered_df.columns and not filtered_df.empty:
        cat_counts = filtered_df['pred_category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        if cat_counts.empty:
            st.info("No data.")
        else:
            fig_pie = px.pie(cat_counts, names='Category', values='Count', hole=0.3,
                             color='Category', color_discrete_sequence=px.colors.sequential.Viridis)
            fig_pie.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                                  margin=dict(t=0,b=0,l=0,r=0), height=200)
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No data.")

# Action Timeline
with row2[2]:
    st.subheader('Action Timeline')
    if {'action','timestamp'}.issubset(filtered_df.columns) and not filtered_df.empty:
        filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
        if filtered_df['action'].dropna().empty:  # no action records
            st.info("No data.")
        else:
            fig_action_timeline = px.histogram(
                filtered_df,
                x='timestamp',
                color='action',
                color_discrete_sequence=px.colors.qualitative.Dark24,
                nbins=30
            )
            fig_action_timeline.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="center", x=0.5),
                margin=dict(t=0,b=0,l=0,r=0),
                height=200
            )
            fig_action_timeline.update_xaxes(title_text="")
            st.plotly_chart(fig_action_timeline, use_container_width=True)
    else:
        st.info("No data.")

# ========== ROW 3 ==========
row3 = st.columns([2,2])

# Top Affected Sites
with row3[0]:
    st.subheader("Top Affected Sites")
    if 'pred_24h' in filtered_df.columns and not filtered_df.empty:
        site_counts = filtered_df[filtered_df['pred_24h']=='Yes'].groupby('site_id').size().reset_index(name='complaints').sort_values('complaints', ascending=False)
        if site_counts.empty:
            st.info("No data.")
        else:
            st.dataframe(site_counts, height=150)
    else:
        st.info("No data.")

# Timeline of Complaints
with row3[1]:
    st.subheader("Timeline of Complaints")
    if 'timestamp' in filtered_df.columns and 'pred_24h' in filtered_df.columns and not filtered_df.empty:
        filtered_yes = filtered_df[filtered_df['pred_24h']=='Yes'].copy()
        if filtered_yes.empty:
            st.info("No data.")
        else:
            filtered_yes['hour'] = pd.to_datetime(filtered_yes['timestamp']).dt.floor('H')
            t_series = filtered_yes.groupby('hour')['pred_24h'].count().reset_index()
            fig_dd_timeline = px.bar(t_series, x='hour', y='pred_24h', labels={'pred_24h':'complaints','hour':'time'}, color='pred_24h',
                                    color_discrete_sequence=px.colors.sequential.Reds)
            fig_dd_timeline.update_layout(height=150, title_text="", showlegend=False, margin=dict(t=0,b=0,l=0,r=0))
            fig_dd_timeline.update_xaxes(title_text="")
            st.plotly_chart(fig_dd_timeline, use_container_width=True)
    else:
        st.info("No data.")

# ========== ROW 4 ==========
row4 = st.columns([4,2])

# Prediction Results Table
with row4[0]:
    st.subheader('Prediction Results Table')
    def highlight_pred(val):
        color = '#8B0000' if val == 'Yes' else 'green'
        return f'background-color: {color}; color: white;'
    if not filtered_df.empty and 'pred_24h' in filtered_df.columns:
        styled_df = filtered_df.style.applymap(highlight_pred, subset=['pred_24h'])
        st.dataframe(styled_df)
    else:
        st.dataframe(filtered_df)
    st.download_button('Download CSV', filtered_df.to_csv(index=False), file_name='vip_complaint_results.csv')


# Interactive Insight
with row4[1]:
    st.subheader("Interactive Insight")

    if 'pred_24h' in filtered_df.columns and not filtered_df.empty:
        yes_count = (filtered_df['pred_24h'] == 'Yes').sum()
        no_count = (filtered_df['pred_24h'] == 'No').sum()
        total = yes_count + no_count

        if total == 0:
            st.info("No data.")
            most_affected_site = "N/A"
        else:
            yes_percent = int((yes_count / total) * 100)

            # Gauge
            fig_yesno = go.Figure(go.Indicator(
                mode="gauge+number",
                value=yes_percent,
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#8B0000"},
                    'bgcolor': "green",
                    'steps': [
                        {'range': [0, yes_percent], 'color': "#8B0000"},
                        {'range': [yes_percent, 100], 'color': "green"},
                    ],
                },
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig_yesno.update_traces(gauge={'shape': "angular"})
            fig_yesno.update_layout(height=90, margin=dict(t=0, b=5, l=0, r=0), template="plotly_dark")
            st.plotly_chart(fig_yesno, use_container_width=True)

            # Most affected site
            yes_sites = filtered_df[filtered_df['pred_24h'] == 'Yes']['site_id']
            most_affected_site = yes_sites.mode()[0] if not yes_sites.mode().empty else "N/A"

            # Top sectors
            if most_affected_site != "N/A" and 'sector' in filtered_df.columns:
                sectors = filtered_df.loc[filtered_df['site_id'] == most_affected_site, 'sector'].value_counts()
                top_sectors = "/".join(sectors[sectors == sectors.max()].index.astype(str)) if not sectors.empty else "N/A"
            else:
                top_sectors = "N/A"

            # Top bands
            if most_affected_site != "N/A" and 'band' in filtered_df.columns and top_sectors != "N/A":
                bands = filtered_df.loc[
                    (filtered_df['site_id'] == most_affected_site) &
                    (filtered_df['sector'].isin(sectors[sectors == sectors.max()].index)),
                    'band'
                ].value_counts()
                top_bands = "/".join(bands[bands == bands.max()].index.astype(str)) if not bands.empty else "N/A"
            else:
                top_bands = "N/A"

            # Top weather
            if most_affected_site != "N/A" and 'weather' in filtered_df.columns:
                weathers = filtered_df.loc[filtered_df['site_id'] == most_affected_site, 'weather'].value_counts()
                top_weather = weathers.idxmax() if not weathers.empty else "N/A"
            else:
                top_weather = "N/A"

            # Top event
            if most_affected_site != "N/A" and 'event_type' in filtered_df.columns:
                events = filtered_df.loc[filtered_df['site_id'] == most_affected_site, 'event_type'].dropna().value_counts()
                top_event = events.idxmax() if not events.empty else None
            else:
                top_event = None

            # Top root cause
            if 'root_cause' in filtered_df.columns and not filtered_df['root_cause'].dropna().empty:
                rc_counts = filtered_df['root_cause'].value_counts()
                top_root = rc_counts.idxmax() if not rc_counts.empty else "N/A"
            else:
                top_root = "N/A"

            # Dominant category
            if 'pred_category' in filtered_df.columns and not filtered_df['pred_category'].dropna().empty:
                dominant_category = filtered_df['pred_category'].mode()[0] if not filtered_df['pred_category'].mode().empty else "N/A"
            else:
                dominant_category = "N/A"

            # Sub root cause for most affected site
            if most_affected_site != "N/A" and 'sub_root_cause' in filtered_df.columns:
                sub_series = filtered_df.loc[filtered_df['site_id'] == most_affected_site, 'sub_root_cause'].dropna()
                if not sub_series.empty:
                    sub_counts = sub_series.str.replace(r"\(.*\)", "", regex=True).str.strip().value_counts()
                    sub_root_cause_most_affected_site = sub_counts.idxmax() if not sub_counts.empty else "N/A"
                else:
                    sub_root_cause_most_affected_site = "N/A"
            else:
                sub_root_cause_most_affected_site = "N/A"

            # Summary Report
            st.markdown(f"""
            **Summary Report**
            - Total records analyzed: **{total}**
            - Predicted complaints (Yes): **{yes_count} ({yes_percent}%)**
            - Most affected site: **{most_affected_site}**
            - Dominant root cause: **{top_root}**
            - Immediate focus: **{sub_root_cause_most_affected_site}**
            """)

            # Narrative Insight
            st.info(
                f"⚠️ AI predicts **{yes_percent}%** of future complaints will be **{top_root}** issues. "
                f"Check **{most_affected_site}** sector **{top_sectors}** band **{top_bands}**, "
                f"focusing on **{sub_root_cause_most_affected_site}** and related KPIs. "
                f"This may affect **{dominant_category}** performance during peak hours, "
                f"especially under **{top_weather}** weather"
                + (f" and event **{top_event}**." if top_event else ".")
                + f" To prevent future complaints."
            )

    else:
        st.info("No data.")

# ========== FOOTER ==========
st.caption(f"Copyright © {datetime.datetime.now().year} "
           f"[Damelia](https://www.youtube.com/@CERDASBersamaDamelia). All Rights Reserved.")
