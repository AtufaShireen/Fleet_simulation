# utils/data_generation.py
import osmnx as ox
import random
from models.driver import Driver
import h3
from shapely.geometry import Point
import pickle
import os


def load_city_graph(city_name: str):
    filename = f"{city_name.replace(', ', '_').replace(' ', '_')}_graph.pkl"
    
    if os.path.exists(filename):
        print(f"📦 Loading pre-saved graph from {filename}...")
        with open(filename, "rb") as f:
            G = pickle.load(f)
    else:
        print(f"🌐 Downloading and saving road network for {city_name}...")
        G = ox.graph_from_place(city_name, network_type="drive")
        G = ox.project_graph(G)
        with open(filename, "wb") as f:
            pickle.dump(G, f)
        print(f"✅ Saved graph to {filename}")
    
    return G


def get_hotspot_nodes(G, num_hotspots=200, seed=42, h3_resolution=9):
    print(f"📌 Generating {num_hotspots} random hotspot nodes...")
    random.seed(seed)
    crs = G.graph["crs"]
    chosen_nodes = random.sample(list(G.nodes), num_hotspots)
    hotspots = []
    for n in chosen_nodes:
        x, y = G.nodes[n]["x"], G.nodes[n]["y"]
        lon, lat = ox.projection.project_geometry(Point(x, y), crs=crs, to_crs="epsg:4326")[0].coords[0]
        h3_cell = h3.latlng_to_cell(lat, lon, res=h3_resolution)
        hotspots.append({
            "node": n,
            "x": x,
            "y": y,
            "lat": lat,
            "lon": lon,
            "h3_cell": h3_cell
        })
    return hotspots


def initialize_drivers(G, num_drivers, hotspot_nodes=None, seed=42, h3_resolution=9, config=None):
    print(f"🚗 Initializing {num_drivers} drivers...")
    config = config or {}
    crs = G.graph["crs"]
    random.seed(seed)
    chosen_nodes = random.sample(list(G.nodes), num_drivers)

    if hotspot_nodes is not None:
        hotspot_node_ids = set(h["node"] for h in hotspot_nodes)
        overlap = set(chosen_nodes).intersection(hotspot_node_ids)
        print(f"🔍 Drivers placed on hotspots: {len(overlap)} (e.g., {list(overlap)[:5]})")

    drivers = []
    for i, n in enumerate(chosen_nodes):
        x, y = G.nodes[n]["x"], G.nodes[n]["y"]
        lon, lat = ox.projection.project_geometry(Point(x, y), crs=crs, to_crs="epsg:4326")[0].coords[0]
        h3_cell = h3.latlng_to_cell(lat, lon, res=h3_resolution)
        print("For Driber", i, "node", n, "location:", (x, y), "lat/lon:", (lat, lon)," h3_cell:", h3_cell)
        drivers.append(Driver(
            driver_id=i,
            initial_location=(x, y),
            vehicle_id=i,
            lat=lat,
            lon=lon,
            h3_cell=h3_cell,
            h3_resolution=h3_resolution,
            config=config
        ))
    return drivers