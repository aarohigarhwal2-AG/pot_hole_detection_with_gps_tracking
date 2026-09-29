"""
map_generator.py - Interactive OpenStreetMap and HeatMap generator using Folium.
"""
import folium
from folium.plugins import HeatMap, MarkerCluster
import os


def generate_map(potholes_data: list, center_lat: float = 28.6139, center_lon: float = 77.2090, zoom: int = 14) -> folium.Map:
    if potholes_data:
        center_lat = potholes_data[0]["latitude"]
        center_lon = potholes_data[0]["longitude"]

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="OpenStreetMap")
    marker_cluster = MarkerCluster(name="Pothole Clusters").add_to(fmap)
    heat_data = []

    for item in potholes_data:
        lat = item["latitude"]
        lon = item["longitude"]
        severity = item["severity"]
        conf = item["confidence"]
        tstamp = item["timestamp"]
        img_path = item.get("image_path", "")

        heat_data.append([lat, lon, 1.0 if severity == "Severe" else (0.6 if severity == "Moderate" else 0.3)])

        color_map = {
            "Minor": "green",
            "Moderate": "orange",
            "Severe": "red"
        }
        marker_color = color_map.get(severity, "blue")

        popup_html = f"""
        <div style='width: 200px; font-family: sans-serif;'>
            <h4 style='margin: 0; color: #333;'>Pothole #{item['pothole_uid']}</h4>
            <p style='margin: 4px 0;'><b>Severity:</b> <span style='color:{marker_color}; font-weight:bold;'>{severity}</span></p>
            <p style='margin: 4px 0;'><b>Confidence:</b> {conf * 100:.1f}%</p>
            <p style='margin: 4px 0;'><b>GPS:</b> {lat:.5f}, {lon:.5f}</p>
            <p style='margin: 4px 0; font-size: 11px; color: #666;'>{tstamp}</p>
        """
        if os.path.exists(img_path):
            popup_html += f"<img src='{img_path}' style='width: 100%; border-radius: 4px; margin-top: 5px;'/>"
        popup_html += "</div>"

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{severity} Pothole ({conf * 100:.1f}%)",
            icon=folium.Icon(color=marker_color, icon="exclamation-sign")
        ).add_to(marker_cluster)

    if heat_data:
        HeatMap(heat_data, radius=15, blur=10, max_zoom=16, name="Pothole Density Heatmap").add_to(fmap)

    folium.LayerControl().add_to(fmap)
    return fmap
