"""
app.py - Streamlit Web Dashboard for Real-time Pothole Monitoring & Fleet Geolocation.
"""
import streamlit as st
import pandas as pd
import os
from streamlit_folium import folium_static
from db_manager import DatabaseManager
from map_generator import generate_map

st.set_page_config(page_title="Pothole AI & GPS Fleet Tracker", layout="wide", page_icon="🛣️")

st.title("🛣️ Intelligent Pothole Detection & GPS Fleet Management")
st.markdown("Automated road condition surveillance combining computer vision with real-time geospatial tagging.")

db = DatabaseManager()

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

potholes = db.get_all_potholes()
total_potholes = len(potholes)
severe_count = sum(1 for p in potholes if p["severity"] == "Severe")
moderate_count = sum(1 for p in potholes if p["severity"] == "Moderate")
minor_count = sum(1 for p in potholes if p["severity"] == "Minor")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Incidents", total_potholes)
c2.metric("Severe Potholes 🔴", severe_count)
c3.metric("Moderate Potholes 🟠", moderate_count)
c4.metric("Minor Bumps 🟢", minor_count)

tab_map, tab_incidents, tab_live = st.tabs(["🗺️ Geographic Map & Heatmap", "📋 Incident Database", "📹 Live Stream Instructions"])

with tab_map:
    st.subheader("Geospatial Incident Map")
    if potholes:
        fmap = generate_map(potholes)
        folium_static(fmap, width=1050, height=520)
    else:
        st.info("No potholes recorded yet. Run generate_demo_data.py or detect_video.py to populate.")

with tab_incidents:
    st.subheader("Recorded Incident Log")
    if potholes:
        df = pd.DataFrame(potholes)
        st.dataframe(df[["id", "pothole_uid", "timestamp", "latitude", "longitude", "confidence", "severity"]], use_container_width=True)

        st.subheader("Snapshot Gallery")
        cols = st.columns(4)
        for i, row in enumerate(potholes[:8]):
            img_path = row.get("image_path", "")
            if os.path.exists(img_path):
                cols[i % 4].image(img_path, caption=f"#{row['pothole_uid']} | {row['severity']} | {row['timestamp']}", use_container_width=True)

        if os.path.exists("pothole_log.csv"):
            with open("pothole_log.csv", "rb") as f:
                st.download_button("📥 Download Full CSV Report", f, file_name="pothole_survey_report.csv", mime="text/csv")
    else:
        st.write("No incident logs available.")

with tab_live:
    st.subheader("Live Detection Instructions")
    st.markdown("""
    ```bash
    # 1. Generate test video
    python generate_demo_data.py

    # 2. Run detection using trained model weights
    python detect_video.py --source demo_road.mp4 --model weights/best.pt --conf 0.48
    ```
    """)
