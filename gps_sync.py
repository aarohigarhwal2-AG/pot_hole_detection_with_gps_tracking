"""
gps_sync.py
-----------
Utilities to turn a GPS log recorded alongside the dashcam video into a
lookup you can query by elapsed video time, so every detected pothole frame
can be tagged with a real-world (latitude, longitude).

WHY THIS IS NEEDED:
A plain dashcam video file has no GPS info baked in. On a bus, GPS usually
comes from one of:
  1. A phone GPS-logger app (GPX/KML/CSV export) running at the same time
     as the dashcam recording.
  2. A dedicated GPS module (u-blox NEO-6M etc.) logging NMEA sentences to
     a serial port / SD card with its own timestamps.
  3. A combined dashcam+GPS unit that already embeds GPS in the video
     metadata or a sidecar .srt/.gpx file (common in car dashcams) —
     if you have this, skip straight to `load_gps_csv`/`load_gpx`.

This module supports:
  - GPX files (most GPS logger apps export this)
  - A simple CSV with columns: timestamp, lat, lon
  - NMEA log files (GPGGA/GPRMC sentences) with a timestamp per line

You give it (a) the log file and (b) the real-world start time of the video
recording (so frame N's "elapsed seconds" can be mapped to an absolute
timestamp, which is then matched/interpolated against the GPS log).
"""

import csv
import re
from bisect import bisect_left
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET


class GPSTrack:
    """Sorted list of (timestamp, lat, lon) with interpolation lookup."""

    def __init__(self, points):
        # points: list of (datetime, lat, lon), sorted ascending
        self.points = sorted(points, key=lambda p: p[0])
        self._times = [p[0] for p in self.points]

    def lookup(self, ts):
        """Return interpolated (lat, lon) for a given datetime `ts`."""
        if not self.points:
            return None, None

        if ts <= self._times[0]:
            return self.points[0][1], self.points[0][2]
        if ts >= self._times[-1]:
            return self.points[-1][1], self.points[-1][2]

        i = bisect_left(self._times, ts)
        t0, lat0, lon0 = self.points[i - 1]
        t1, lat1, lon1 = self.points[i]

        span = (t1 - t0).total_seconds()
        frac = 0.0 if span == 0 else (ts - t0).total_seconds() / span

        lat = lat0 + (lat1 - lat0) * frac
        lon = lon0 + (lon1 - lon0) * frac
        return lat, lon


def load_gpx(path):
    """Parse a GPX track file into a GPSTrack."""
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    tree = ET.parse(path)
    root = tree.getroot()

    points = []
    for trkpt in root.findall(".//g:trkpt", ns):
        lat = float(trkpt.attrib["lat"])
        lon = float(trkpt.attrib["lon"])
        time_el = trkpt.find("g:time", ns)
        if time_el is None:
            continue
        ts = datetime.fromisoformat(time_el.text.replace("Z", "+00:00"))
        points.append((ts, lat, lon))

    return GPSTrack(points)


def load_gps_csv(path, timestamp_fmt="%Y-%m-%d %H:%M:%S"):
    """
    Parse a CSV with columns: timestamp, lat, lon
    e.g.
        timestamp,lat,lon
        2026-09-03 08:15:00,21.2100,81.3800
        2026-09-03 08:15:01,21.2101,81.3802
    """
    points = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = datetime.strptime(row["timestamp"].strip(), timestamp_fmt)
            points.append((ts, float(row["lat"]), float(row["lon"])))
    return GPSTrack(points)


_NMEA_TIME_RE = re.compile(r"\$GP(GGA|RMC),(\d{6}\.?\d*),")


def _nmea_to_deg(coord, direction):
    """Convert NMEA ddmm.mmmm format to decimal degrees."""
    if not coord:
        return None
    dot = coord.find(".")
    deg_len = dot - 2
    deg = float(coord[:deg_len])
    minutes = float(coord[deg_len:])
    dec = deg + minutes / 60.0
    if direction in ("S", "W"):
        dec = -dec
    return dec


def load_nmea_log(path, ref_date):
    """
    Parse a raw NMEA log (GPGGA sentences) into a GPSTrack.
    `ref_date`: a `date` object for the day the log was recorded, since NMEA
    time fields only carry HHMMSS (no date).
    """
    points = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "$GPGGA" not in line and "$GNGGA" not in line:
                continue
            parts = line.strip().split(",")
            if len(parts) < 6 or not parts[1] or not parts[2]:
                continue
            try:
                hh = int(parts[1][0:2])
                mm = int(parts[1][2:4])
                ss = float(parts[1][4:])
                lat = _nmea_to_deg(parts[2], parts[3])
                lon = _nmea_to_deg(parts[4], parts[5])
            except (ValueError, IndexError):
                continue
            if lat is None or lon is None:
                continue
            ts = datetime(ref_date.year, ref_date.month, ref_date.day,
                           hh, mm, int(ss), int((ss % 1) * 1e6))
            points.append((ts, lat, lon))
    return GPSTrack(points)


def video_time_to_timestamp(video_start_time, elapsed_seconds):
    """elapsed_seconds = frame_number / fps"""
    return video_start_time + timedelta(seconds=elapsed_seconds)
