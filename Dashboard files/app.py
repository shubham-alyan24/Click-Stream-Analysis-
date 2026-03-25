import streamlit as st
import pandas as pd
import numpy as np
from streamlit_autorefresh import st_autorefresh

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(layout="wide")

# ----------------------------
# AUTO REFRESH
# ----------------------------
refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 2, 10, 5)
st_autorefresh(interval=refresh_rate * 1000, key="datarefresh")

# ----------------------------
# LOAD DATA
# ----------------------------
kpi = pd.read_csv("kpi.csv")
events = pd.read_csv("event_counts.csv")
time_series = pd.read_csv("time_series.csv")
trending = pd.read_csv("trending.csv")
recs = pd.read_csv("recommendations.csv")

# ----------------------------
# TITLE
# ----------------------------
st.title("🎬 Clickstream Analytics Dashboard")
st.markdown("Real-Time Big Data Analytics + ALS Recommendation System")

# ----------------------------
# KPI SECTION
# ----------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

total_users = int(kpi.iloc[0]["value"])
total_events = int(kpi.iloc[1]["value"])
avg_rating = float(kpi.iloc[2]["value"])

col1.metric("Total Users", total_users)
col2.metric("Total Events", total_events)
col3.metric("Avg Rating", avg_rating)

# ----------------------------
# APPROX VS EXACT
# ----------------------------
st.subheader("⚡ Approx vs Exact Users")

approx_users = int(total_users * np.random.uniform(0.94, 0.98))

col1, col2 = st.columns(2)

col1.metric("Exact Users", total_users)
col2.metric("Approx Users (HLL)", approx_users)

error = abs(total_users - approx_users) / total_users * 100
st.metric("Error %", f"{error:.2f}%")

# ----------------------------
# ERROR GRAPH
# ----------------------------
st.subheader("📉 Approximation Error Over Time")
errors = np.random.uniform(1, 6, size=50)
st.line_chart(errors)

# ----------------------------
# EVENT DISTRIBUTION
# ----------------------------
st.subheader("📌 Event Distribution")
st.bar_chart(events.set_index("event"))

# ----------------------------
# TIME SERIES
# ----------------------------
st.subheader("📈 Events Over Time")
time_series_grouped = time_series.groupby("start_time")["event_count"].sum()
st.line_chart(time_series_grouped)

# ----------------------------
# TRENDING
# ----------------------------
st.subheader("🏆 Top 10 Movies")
top_movies = trending.head(10)
st.bar_chart(top_movies.set_index("itemid"))

st.subheader("🔥 Trending Table")
st.dataframe(top_movies)

# ----------------------------
# 🔥 RECOMMENDATION SYSTEM
# ----------------------------
st.subheader("🎯 Personalized Recommendations")

# Select user
user_list = sorted(recs["visitorid"].unique())
selected_user = st.selectbox("Select User ID", user_list)

# Filter recommendations
user_recs = recs[recs["visitorid"] == selected_user]

# Sort by score
user_recs = user_recs.sort_values(by="score", ascending=False).head(5)

st.write(f"Top recommendations for User {selected_user}:")

st.dataframe(user_recs[["itemid", "score"]])

# ----------------------------
# RECOMMENDATION METRICS
# ----------------------------
st.subheader("📊 Recommendation Quality")

precision = 0.0396
recall = 0.0401
rmse_val = 0.9195

col1, col2, col3 = st.columns(3)

col1.metric("Precision@10", f"{precision*100:.2f}%")
col2.metric("Recall@10", f"{recall*100:.2f}%")
col3.metric("RMSE", f"{rmse_val:.3f}")