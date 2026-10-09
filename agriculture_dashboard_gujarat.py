from pathlib import Path
import glob
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from folium.plugins import Fullscreen

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "visualizations"
OUT = ROOT / "agriculture_dashboard_output"
OUT.mkdir(exist_ok=True)


def load_summary(folder):
    """Load and combine Spark part CSV files from a summary folder."""
    files = sorted(glob.glob(str(DATA / folder / "part-*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No part-*.csv found in {DATA / folder}. "
            "Export the corresponding HDFS summary first."
        )
    return pd.concat(
        [pd.read_csv(file) for file in files],
        ignore_index=True,
    )


def require_columns(frame, required, label):
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(
            f"{label} summary is missing columns: {', '.join(sorted(missing))}"
        )


# Read the five small Spark summaries, not the original 1.14 GB CSV.
crop = load_summary("crop_yield")
irrigation = load_summary("irrigation")
geo = load_summary("geographic")
temperature = load_summary("temperature")
soil = load_summary("soil")

require_columns(crop, ["crop", "average_yield_kg"], "Crop yield")
require_columns(
    irrigation,
    [
        "irrigation_status",
        "irrigation_method",
        "average_water_usage_liters",
        "average_yield_kg",
    ],
    "Irrigation",
)
require_columns(
    geo,
    [
        "location",
        "crop",
        "record_count",
        "average_yield_kg",
        "latitude",
        "longitude",
    ],
    "Geographic",
)
require_columns(
    temperature,
    ["temperature_range", "average_temperature", "average_yield_kg"],
    "Temperature",
)
require_columns(soil, ["crop", "soil_ph_range", "average_yield_kg"], "Soil")

# 1. Crop-wise average expected yield
crop_plot = crop.sort_values("average_yield_kg", ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(crop_plot["crop"], crop_plot["average_yield_kg"])
ax.set_title("Average Expected Yield by Crop")
ax.set_xlabel("Crop")
ax.set_ylabel("Average expected yield (kg)")
ax.tick_params(axis="x", rotation=20)
ax.bar_label(bars, fmt="%.0f", padding=3)
fig.tight_layout()
fig.savefig(OUT / "01_crop_yield.png", dpi=160)
plt.close(fig)

# 2. Temperature range vs average expected yield
temperature_plot = temperature.sort_values("average_temperature")
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(
    temperature_plot["temperature_range"].astype(str),
    temperature_plot["average_yield_kg"],
)
ax.set_title("Average Expected Yield by Temperature Range")
ax.set_xlabel("Temperature range")
ax.set_ylabel("Average expected yield (kg)")
ax.tick_params(axis="x", rotation=15)
ax.bar_label(bars, fmt="%.0f", padding=3)
fig.tight_layout()
fig.savefig(OUT / "02_temperature_yield.png", dpi=160)
plt.close(fig)

# 3. Recorded water usage by irrigation method and status.
# This is descriptive summary data, not proof of causal effects.
water = irrigation.groupby(
    ["irrigation_status", "irrigation_method"],
    as_index=False,
).agg(
    average_water_usage_liters=("average_water_usage_liters", "mean"),
    average_yield_kg=("average_yield_kg", "mean"),
)
pivot = water.pivot(
    index="irrigation_method",
    columns="irrigation_status",
    values="average_water_usage_liters",
).fillna(0)

fig, ax = plt.subplots(figsize=(10, 6))
pivot.plot(kind="bar", ax=ax)
ax.set_title("Recorded Average Water Usage by Irrigation Method and Status")
ax.set_xlabel("Irrigation method")
ax.set_ylabel("Average water usage (liters)")
ax.tick_params(axis="x", rotation=15)
ax.legend(title="Irrigation status")
fig.tight_layout()
fig.savefig(OUT / "03_irrigation_water_usage.png", dpi=160)
plt.close(fig)

# 4. Soil pH range vs average expected yield, grouped by crop
fig, ax = plt.subplots(figsize=(12, 7))
for crop_name, group in soil.groupby("crop"):
    group = group.sort_values("soil_ph_range")
    ax.plot(
        group["soil_ph_range"].astype(str),
        group["average_yield_kg"],
        marker="o",
        label=crop_name,
    )
ax.set_title("Average Expected Yield by Crop and Soil pH Range")
ax.set_xlabel("Soil pH range")
ax.set_ylabel("Average expected yield (kg)")
ax.tick_params(axis="x", rotation=20)
ax.legend(title="Crop")
fig.tight_layout()
fig.savefig(OUT / "04_soil_ph_yield.png", dpi=160)
plt.close(fig)

# 5. Interactive Gujarat map using a local GeoJSON boundary.
# No external map tiles or tile-provider API key are used.
boundary_file = DATA / "gujarat.geojson"
if not boundary_file.exists():
    raise FileNotFoundError(
        f"Gujarat boundary file not found: {boundary_file}\n"
        "Download it in PowerShell with:\n"
        'Invoke-WebRequest -Uri "https://raw.githubusercontent.com/udit-001/india-maps-data/main/geojson/states/gujarat.geojson" '
        '-OutFile ".\\visualizations\\gujarat.geojson"'
    )

with boundary_file.open("r", encoding="utf-8") as file:
    gujarat_boundary = json.load(file)

geo = geo.copy()
geo["latitude"] = pd.to_numeric(geo["latitude"], errors="coerce")
geo["longitude"] = pd.to_numeric(geo["longitude"], errors="coerce")
geo["average_yield_kg"] = pd.to_numeric(
    geo["average_yield_kg"], errors="coerce"
)
geo["record_count"] = pd.to_numeric(geo["record_count"], errors="coerce")
geo = geo.dropna(
    subset=["latitude", "longitude", "average_yield_kg", "record_count"]
)

if geo.empty:
    raise ValueError("The geographic summary has no valid coordinate rows.")

map_center = [
    float(geo["latitude"].mean()),
    float(geo["longitude"].mean()),
]
m = folium.Map(location=map_center, zoom_start=7, tiles=None)
Fullscreen().add_to(m)

folium.GeoJson(
    gujarat_boundary,
    name="Gujarat state boundary",
    style_function=lambda feature: {
        "fillColor": "#9ac7a5",
        "color": "#176b36",
        "weight": 2,
        "fillOpacity": 0.20,
    },
    tooltip="Gujarat state boundary",
).add_to(m)

for _, row in geo.iterrows():
    yield_value = float(row["average_yield_kg"])
    popup_html = (
        f"<b>Location:</b> {row['location']}<br>"
        f"<b>Crop:</b> {row['crop']}<br>"
        f"<b>Records:</b> {int(row['record_count']):,}<br>"
        f"<b>Average expected yield:</b> {yield_value:,.2f} kg<br>"
        f"<b>Latitude:</b> {float(row['latitude']):.5f}<br>"
        f"<b>Longitude:</b> {float(row['longitude']):.5f}"
    )
    folium.CircleMarker(
        location=[
            float(row["latitude"]),
            float(row["longitude"]),
        ],
        radius=max(5, min(14, yield_value / 1200)),
        popup=folium.Popup(popup_html, max_width=320),
        tooltip=(
            f"{row['location']} - {row['crop']}: "
            f"{yield_value:,.0f} kg"
        ),
        color="#175cd3",
        fill=True,
        fill_color="#3987f5",
        fill_opacity=0.8,
        weight=1,
    ).add_to(m)

folium.LayerControl().add_to(m)

# Fit map bounds to the Gujarat boundary's GeoJSON coordinates.
# fit_bounds on markers only would crop the full state outline, so use
# a reasonable geographic extent around the state and dataset locations.
m.fit_bounds([[20.0, 68.0], [24.8, 74.5]])
m.save(str(OUT / "05_gujarat_agriculture_map.html"))

# 6. HTML dashboard
html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agriculture Telemetry - Spark Analytics Dashboard</title>
<style>
body{font-family:Arial,sans-serif;margin:0;background:#f4f6f8;color:#17202a}
header{background:#204d36;color:white;padding:24px}
main{max-width:1200px;margin:20px auto;padding:0 16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}
section{background:white;padding:16px;border-radius:10px;box-shadow:0 2px 8px #0001;min-width:0}
img{width:100%;height:auto}
iframe{width:100%;height:620px;border:0}
a{color:#176b43}.note{font-size:.92rem;color:#4b5563}
</style>
</head>
<body>
<header>
<h1>Agriculture Telemetry Analytics</h1>
<p>Summary analytics generated from Apache Spark output on HDFS</p>
</header>
<main>
<p class="note">
These charts summarize exported aggregations. Relationships are descriptive,
not proof of causation. Map positions use coordinates stored in the dataset.
The map uses a local Gujarat GeoJSON boundary and no external map tiles.
Folium's JavaScript/CSS resources may still require internet access.
</p>
<div class="grid">
<section><h2>Crop-wise yield</h2><img src="01_crop_yield.png" alt="Average expected yield by crop"></section>
<section><h2>Temperature and yield</h2><img src="02_temperature_yield.png" alt="Average expected yield by temperature range"></section>
<section><h2>Irrigation and water usage</h2><img src="03_irrigation_water_usage.png" alt="Average water usage by irrigation method and status"></section>
<section><h2>Soil pH and yield</h2><img src="04_soil_ph_yield.png" alt="Average expected yield by crop and soil pH range"></section>
</div>
<section style="margin-top:18px">
<h2>Interactive Gujarat location map</h2>
<iframe src="05_gujarat_agriculture_map.html" title="Interactive agricultural location map"></iframe>
<p><a href="05_gujarat_agriculture_map.html" target="_blank">Open map in a separate tab</a></p>
</section>
</main>
</body>
</html>
"""
(OUT / "index.html").write_text(html, encoding="utf-8")
(OUT / "README.txt").write_text(
    "Agriculture Telemetry Dashboard\n\n"
    "1. Keep the visualizations folder beside this script.\n"
    "2. Ensure visualizations/gujarat.geojson exists.\n"
    "3. Run: python agriculture_dashboard_gujarat.py\n"
    "4. Open agriculture_dashboard_output/index.html in a browser.\n\n"
    "Required summary folders: crop_yield, irrigation, geographic, "
    "temperature, soil.\n",
    encoding="utf-8",
)

print("Dashboard generation completed.")
print(f"Open: {OUT / 'index.html'}")
print("Generated files:")
for path in sorted(OUT.iterdir()):
    print(f" - {path.name} ({path.stat().st_size:,} bytes)")
