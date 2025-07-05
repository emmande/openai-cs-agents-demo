import streamlit as st
import sqlite3
import pandas as pd
import altair as alt

DB_NAME = "agent_events.db"

def fetch_logs_df():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM agent_events", conn)
    conn.close()
    return df

st.title("Conversation Log Visualizations")

df = fetch_logs_df()

if df.empty:
    st.info("No data available.")
else:
    # Convert log_datetime to datetime
    df['log_datetime_dt'] = pd.to_datetime(df['log_datetime'], errors='coerce')

    tab1, tab2, tab3 = st.tabs([
        "Basic Visualizations",
        "Combined Breakdown",
        "Plot Over Time"
    ])

    with tab1:
        # 1. Unique convID count
        unique_conv_count = df["convID"].nunique()
        st.metric("Total Unique Conversation IDs", unique_conv_count)

        # 2. Breakdown by agent_name
        agent_count = (
            df.groupby("agent_name")["convID"]
            .nunique()
            .reset_index()
            .rename(columns={"convID": "Unique Conversations"})
        )
        st.subheader("Breakdown by Agent Name")
        chart_agent = alt.Chart(agent_count).mark_bar().encode(
            x=alt.X("agent_name", title="Agent Name"),
            y=alt.Y("Unique Conversations"),
            tooltip=["agent_name", "Unique Conversations"]
        )
        st.altair_chart(chart_agent, use_container_width=True)

        # 3. Breakdown by flag_miss
        flag_count = (
            df.groupby("flag_miss")["convID"]
            .nunique()
            .reset_index()
            .rename(columns={"convID": "Unique Conversations"})
        )
        st.subheader("Breakdown by Flag Miss")
        chart_flag = alt.Chart(flag_count).mark_bar().encode(
            x=alt.X("flag_miss", title="Flag Miss"),
            y=alt.Y("Unique Conversations"),
            tooltip=["flag_miss", "Unique Conversations"]
        )
        st.altair_chart(chart_flag, use_container_width=True)

    with tab2:
        # Combined breakdown: flag_miss by agent_name
        combined = (
            df.groupby(["agent_name", "flag_miss"])["convID"]
            .nunique()
            .reset_index()
            .rename(columns={"convID": "Unique Conversations"})
        )
        st.subheader("Flag Miss by Agent Name (Unique Conversation Count)")
        chart = alt.Chart(combined).mark_bar().encode(
            x=alt.X("agent_name:N", title="Agent Name"),
            y=alt.Y("Unique Conversations:Q"),
            color=alt.Color("flag_miss:N", title="Flag Miss"),
            tooltip=["agent_name", "flag_miss", "Unique Conversations"]
        )
        st.altair_chart(chart, use_container_width=True)

    with tab3:

        st.subheader("Unique Conversations Over Time (by Agent and Flag Miss)")
        df_time = df.dropna(subset=['log_datetime_dt'])
        df_time['date'] = df_time['log_datetime_dt'].dt.date

        # Group by date, agent_name, flag_miss
        time_breakdown = (
            df_time.groupby(['date', 'agent_name', 'flag_miss'])['convID']
            .nunique()
            .reset_index()
            .rename(columns={"convID": "Unique Conversations"})
        )

        # Altair line chart: color by agent_name, facet by flag_miss
        chart_time = alt.Chart(time_breakdown).mark_line(point=True).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("Unique Conversations:Q"),
            color=alt.Color("agent_name:N", title="Agent Name"),
            tooltip=["date", "agent_name", "flag_miss", "Unique Conversations"]
        ).facet(
            column=alt.Column("flag_miss:N", title="Flag Miss")
        ).resolve_scale(
            y='independent'
        )

        st.altair_chart(chart_time, use_container_width=True)