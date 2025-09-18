import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import networkx as nx  # New: For Dijkstra
from math import radians, cos, sin, asin, sqrt
import folium
from streamlit_folium import st_folium
from dotenv import load_dotenv

# Load env
load_dotenv()
MAPBOX_API_KEY = os.getenv("API_MAP", "")

# Paths
BUS_PATH = "PMPL dataset.xlsx"
METRO_PATH = "pune_metro.xlsx"
ROUTES_PATH = "pune_routes_big_training.csv"
MODEL_PATH = "model/transport_model.pkl"
ENCODER_PATH = "model/label_encoder.pkl"

# Emission rates (g CO2 / km)
EMISSION_RATES = {"Walk": 0, "Metro": 20, "Bus": 100, "Mixed": 60}

# Haversine
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c  # km

# Polyline decoder
def decode_polyline(polyline_str, precision=6):
    index, lat, lng = 0, 0, 0
    coordinates = []
    changes = {'latitude': 0, 'longitude': 0}
    factor = 10 ** precision

    while index < len(polyline_str):
        for unit in ['latitude', 'longitude']:
            shift, result = 0, 0
            while True:
                byte = ord(polyline_str[index]) - 63
                index += 1
                result |= (byte & 0x1f) << shift
                shift += 5
                if byte < 0x20:
                    break
            if (result & 1):
                changes[unit] = ~(result >> 1)
            else:
                changes[unit] = (result >> 1)
        lat += changes['latitude']
        lng += changes['longitude']
        coordinates.append((lat / factor, lng / factor))
    return coordinates

# Get real route from Mapbox
def get_route_geometry(lat1, lon1, lat2, lon2, mode, access_token):
    profile = 'mapbox/walking' if mode == 'Walk' else 'mapbox/driving'
    coordinates = f"{lon1},{lat1};{lon2},{lat2}"
    url = f"https://api.mapbox.com/directions/v5/{profile}/{coordinates}"
    params = {
        'access_token': access_token,
        'geometries': 'polyline6',
        'overview': 'full',
        'steps': 'true'
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if 'routes' in data and data['routes']:
                route = data['routes'][0]
                geometry = route['geometry']
                points = decode_polyline(geometry, precision=6)
                dist_km = route['distance'] / 1000
                duration_min = route['duration'] / 60
                return points, dist_km, duration_min
    except Exception as e:
        st.warning(f"Route API error: {e}. Using straight line.")
    dist = haversine(lon1, lat1, lon2, lat2)
    return [(lat1, lon1), (lat2, lon2)], dist, dist * (15 if mode == 'Walk' else 5)

# Load data
@st.cache_data
def load_data():
    bus_df = pd.read_excel(BUS_PATH, sheet_name="Sheet1")
    metro_df = pd.read_excel(METRO_PATH, sheet_name="pune_metro_full_schedule_with_c")
    routes_df = pd.read_csv(ROUTES_PATH)
    return bus_df, metro_df, routes_df

bus_df, metro_df, routes_df = load_data()

# Load model
model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

# Locations
LOCATIONS = {
    "Kothrud Depot": (18.5081, 73.8089),
    "Swargate": (18.5011, 73.8617),
    "Shivaji Nagar": (18.5309, 73.8470),
    "PCMC": (18.6298, 73.7997),
    "Bhosari": (18.6290, 73.8210)
}

# Find relevant stations (with detour filter)
def find_stations_in_path(orig_lat, orig_lon, dest_lat, dest_lon, buffer_km=1.5):
    num_points = 50
    lats = np.linspace(orig_lat, dest_lat, num_points)
    lons = np.linspace(orig_lon, dest_lon, num_points)
    
    bus_stops = bus_df[["Start Stop", "Start_Lat", "Start_Lon"]].rename(columns={"Start Stop": "Stop", "Start_Lat": "Lat", "Start_Lon": "Lon"})
    bus_stops["Type"] = "Bus"
    metro_stops = metro_df[["Start_Station", "Start_Latitude", "Start_Longitude"]].rename(columns={"Start_Station": "Stop", "Start_Latitude": "Lat", "Start_Longitude": "Lon"})
    metro_stops["Type"] = "Metro"
    all_stops = pd.concat([bus_stops, metro_stops]).drop_duplicates()
    
    total_direct_dist = haversine(orig_lon, orig_lat, dest_lon, dest_lat)
    nearby = []
    for _, stop in all_stops.iterrows():
        min_dist_to_path = min(haversine(stop["Lon"], stop["Lat"], lon, lat) for lat, lon in zip(lats, lons))
        dist_orig_to_stop = haversine(orig_lon, orig_lat, stop["Lon"], stop["Lat"])
        dist_stop_to_dest = haversine(stop["Lon"], stop["Lat"], dest_lon, dest_lat)
        detour_factor = (dist_orig_to_stop + dist_stop_to_dest) / total_direct_dist
        
        if min_dist_to_path <= buffer_km and detour_factor <= 1.2:
            nearby.append(stop)
    
    df_nearby = pd.DataFrame(nearby)
    if not df_nearby.empty:
        df_nearby['dist_from_orig'] = df_nearby.apply(lambda r: haversine(orig_lon, orig_lat, r['Lon'], r['Lat']), axis=1)
        df_nearby = df_nearby.sort_values('dist_from_orig').head(7)
    return df_nearby

# Predict mode
def predict_mode(dist_km, dist_end_km, fare, time_min, co2_save=0):
    features = np.array([[dist_km, dist_end_km, fare, time_min, co2_save]])
    pred = model.predict(features)[0]
    return le.inverse_transform([pred])[0]

# Build graph for Dijkstra and find shortest time path
def find_shortest_path(orig, dest, stations):
    G = nx.Graph()
    all_nodes = ['Origin'] + list(stations['Stop']) + ['Destination']
    G.add_nodes_from(all_nodes)
    
    # Add edges with time weights (heuristic or from haversine)
    coords = {'Origin': (orig[0], orig[1]), 'Destination': (dest[0], dest[1])}
    for _, s in stations.iterrows():
        coords[s['Stop']] = (s['Lat'], s['Lon'])
    
    for i in range(len(all_nodes)):
        for j in range(i+1, len(all_nodes)):
            n1, n2 = all_nodes[i], all_nodes[j]
            lat1, lon1 = coords[n1]
            lat2, lon2 = coords[n2]
            dist = haversine(lon1, lat1, lon2, lat2)
            if dist <= 5:  # Only connect nearby nodes
                time_min = dist * 5  # Heuristic: 5 min/km for transit
                G.add_edge(n1, n2, weight=time_min)
    
    try:
        path = nx.shortest_path(G, source='Origin', target='Destination', weight='weight')
        return path
    except nx.NetworkXNoPath:
        return ['Origin', 'Destination']  # Fallback

# Assemble route from Dijkstra path
def assemble_route(orig, dest, stations, path):
    chain = []
    prev_lat, prev_lon, prev_name = orig[0], orig[1], path[0]
    
    for i, next_name in enumerate(path[1:]):
        if next_name == 'Destination':
            next_lat, next_lon = dest[0], dest[1]
        else:
            next_row = stations[stations['Stop'] == next_name]
            if next_row.empty:
                continue
            next_lat, next_lon = next_row.iloc[0]['Lat'], next_row.iloc[0]['Lon']
        
        # Predict mode for edge
        dist_end = haversine(next_lon, next_lat, dest[1], dest[0])
        co2_save = 50 if next_name in stations[stations['Type']=='Metro']['Stop'].values else 20
        mode = predict_mode(0, dist_end, 10, 10, co2_save)
        if mode == 'Mixed': mode = stations[stations['Stop']==next_name].iloc[0]['Type'] if not stations[stations['Stop']==next_name].empty else 'Bus'
        
        points, dist_km, time_min = get_route_geometry(prev_lat, prev_lon, next_lat, next_lon, mode, MAPBOX_API_KEY)
        fare = 10 if dist_km < 2 else 20
        
        # Override long walks
        if mode == 'Walk' and dist_km > 2:
            mode = 'Bus'
        
        chain.append({
            "from": prev_name, "to": next_name, "mode": mode,
            "points": points, "dist_km": dist_km, "fare": fare, "time_min": time_min,
            "co2_g": dist_km * EMISSION_RATES.get(mode, 0)
        })
        prev_lat, prev_lon, prev_name = next_lat, next_lon, next_name
    
    return chain

# UI
st.set_page_config(layout="wide")
st.title("🌍 Pune Eco-Friendly Route Recommender")

origins = list(LOCATIONS.keys())
dests = list(LOCATIONS.keys())

col1, col2 = st.columns(2)
with col1:
    origin_text = st.selectbox("Select Origin", origins)
with col2:
    dest_text = st.selectbox("Select Destination", dests, index=1 if origin_text == origins[0] else 0)

run_button = st.button("Recommend Route")

# Main logic
if "map_obj" not in st.session_state:
    st.session_state["map_obj"] = folium.Map(location=[18.5204, 73.8567], zoom_start=12, tiles=None)
    folium.TileLayer(
        tiles=f"https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{{z}}/{{x}}/{{y}}?access_token={MAPBOX_API_KEY}",
        attr="Mapbox",
        name="Mapbox Streets"
    ).add_to(st.session_state["map_obj"])

if run_button:
    orig = (*LOCATIONS[origin_text], origin_text)
    dest = (*LOCATIONS[dest_text], dest_text)
    
    stations = find_stations_in_path(orig[0], orig[1], dest[0], dest[1])
    path = find_shortest_path(orig, dest, stations)
    chain = assemble_route(orig, dest, stations, path)
    
    m = folium.Map(location=[(orig[0] + dest[0])/2, (orig[1] + dest[1])/2], zoom_start=13, tiles=None)
    folium.TileLayer(
        tiles=f"https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{{z}}/{{x}}/{{y}}?access_token={MAPBOX_API_KEY}",
        attr="Mapbox",
        name="Mapbox Streets"
    ).add_to(m)
    
    folium.Marker([orig[0], orig[1]], tooltip="Origin", icon=folium.Icon(color="blue")).add_to(m)
    folium.Marker([dest[0], dest[1]], tooltip="Destination", icon=folium.Icon(color="green")).add_to(m)
    
    color_map = {"Walk": "blue", "Metro": "red", "Bus": "green", "Mixed": "purple"}
    total_dist, total_fare, total_time, total_co2 = 0, 0, 0, 0
    
    for seg in chain:
        folium.PolyLine(
            seg['points'],
            color=color_map.get(seg["mode"], "black"), weight=5, tooltip=f"{seg['mode']} ({seg['dist_km']:.2f} km)"
        ).add_to(m)
        total_dist += seg['dist_km']
        total_fare += seg['fare']
        total_time += seg['time_min']
        total_co2 += seg['co2_g']
    
    st.session_state["map_obj"] = m
    
    st.subheader("Recommended Eco-Friendly Route (Shortest Time via Dijkstra)")
    for i, seg in enumerate(chain):
        st.write(f"Segment {i+1}: {seg['from']} → {seg['to']} via {seg['mode']} ({seg['dist_km']:.2f} km, ₹{seg['fare']}, ~{seg['time_min']:.0f} min, {seg['co2_g']:.0f}g CO2)")
    st.write(f"**Total**: {total_dist:.2f} km, ₹{total_fare}, ~{total_time:.0f} min, {total_co2:.0f}g CO2 (Eco-savings: Prioritized low-emission modes)")

# Render map
st_folium(st.session_state["map_obj"], width=900, height=600, returned_objects=[])