"""
map_visualize.py
------------------
Turn potholes.csv (output of detect_and_geotag.py) into an interactive HTML
map with a pin at every pothole location — the kind of thing you can hand
to a municipal/govt road-maintenance office directly.

Usage:
    python map_visualize.py --csv potholes.csv --out potholes_map.html
"""

import argparse
import pandas as pd
import folium
from folium.plugins import MarkerCluster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="potholes_map.html")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["lat", "lon"])

    if df.empty:
        print("No geotagged potholes found in the CSV (lat/lon are empty). "
              "Run detect_and_geotag.py with --gps_file to get coordinates.")
        return

    center = [df["lat"].mean(), df["lon"].mean()]
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        popup = (
            f"<b>Pothole #{int(row['event_id'])}</b><br>"
            f"Confidence: {row['max_confidence']:.2f}<br>"
            f"Seen for {int(row['num_frames_seen'])} frames "
            f"({row['start_sec']}s - {row['end_sec']}s)<br>"
            f"Lat/Lon: {row['lat']:.6f}, {row['lon']:.6f}"
        )
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup, max_width=250),
            icon=folium.Icon(color="red", icon="warning-sign"),
        ).add_to(cluster)

    m.save(args.out)
    print(f"Map with {len(df)} potholes saved -> {args.out}")
    print("Open this HTML file in a browser to view/share it.")


if __name__ == "__main__":
    main()
