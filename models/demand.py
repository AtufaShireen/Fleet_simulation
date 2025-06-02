from typing import List
import random
import networkx as nx
import osmnx as ox
import h3
from shapely.geometry import Point


class DemandUnit:
    def __init__(
        self,
        demand_id: int,
        origin_node: int,
        destination_node: int,
        max_wait_time: int = 50,
        origin_lat: float = None,
        origin_lon: float = None,
        destination_lat: float = None,
        destination_lon: float = None,
        origin_h3_cell: str = None,
        destination_h3_cell: str = None,
    ):
        self.demand_id = demand_id
        self.origin_node = origin_node
        self.destination_node = destination_node
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.destination_lat = destination_lat
        self.destination_lon = destination_lon
        self.origin_h3_cell = origin_h3_cell
        self.destination_h3_cell = destination_h3_cell
        self.wait_time = 0
        self.max_wait_time = max_wait_time
        self.active = True

    def step(self):
        self.wait_time += 1
        if self.wait_time > self.max_wait_time:
            self.active = False


class DemandModel:
    def __init__(self, city_hotspot_nodes: List[dict], G: nx.Graph, max_distance_m: int = 5000, h3_resolution: int = 9):
        print("📦 Initializing Demand Model...")
        self.G = G
        self.hotspot_nodes = [h["node"] for h in city_hotspot_nodes]
        self.hotspot_metadata = {h["node"]: h for h in city_hotspot_nodes}
        self.demand_units: List[DemandUnit] = []
        self.counter = 0
        self.reachable_by_hotspot = {}
        self.max_distance_m = max_distance_m
        self.h3_resolution = h3_resolution
        self.crs = G.graph["crs"]


        print("Preparing reachable destinations for hotspots...")
        for origin in self.hotspot_nodes:
            lengths = nx.single_source_dijkstra_path_length(
                G, origin, cutoff=self.max_distance_m, weight="length"
            )
            destinations = [node for node in lengths if node != origin]
            if destinations:
                self.reachable_by_hotspot[origin] = destinations

        print(f"✅ Reachable destinations prepared for {len(self.reachable_by_hotspot)} hotspots")

    def generate_demand(
        self, G: nx.Graph, num_units: int = 1000, max_distance_m: int = None, reuse_fraction: float = 0.3
    ):
        max_distance_m = max_distance_m or self.max_distance_m
        new_demand_units = []

        # Reuse logic
        num_reuse = int(num_units * reuse_fraction)
        if self.demand_units and num_reuse > 0:
            reused = random.sample(self.demand_units, k=min(num_reuse, len(self.demand_units)))
            for unit in reused:
                new_demand_units.append(
                    DemandUnit(
                        demand_id=self.counter,
                        origin_node=unit.origin_node,
                        destination_node=unit.destination_node,
                        max_wait_time=unit.max_wait_time,
                        origin_lat=unit.origin_lat,
                        origin_lon=unit.origin_lon,
                        destination_lat=unit.destination_lat,
                        destination_lon=unit.destination_lon,
                        origin_h3_cell=unit.origin_h3_cell,
                        destination_h3_cell=unit.destination_h3_cell,
                    )
                )
                self.counter += 1
            print(f"♻️ Reused {len(reused)} past demands")

        # Generate new demands
        while len(new_demand_units) < num_units:
            if not self.reachable_by_hotspot:
                print("⚠️ No precomputed reachable hotspots found.")
                break

            origin = random.choice(list(self.reachable_by_hotspot.keys()))
            destinations = self.reachable_by_hotspot[origin]
            if not destinations:
                continue

            destination = random.choice(destinations)

            # Compute lat/long and h3_cell for origin and destination
            origin_x, origin_y = G.nodes[origin]["x"], G.nodes[origin]["y"]
            dest_x, dest_y = G.nodes[destination]["x"], G.nodes[destination]["y"]
            origin_lon, origin_lat = ox.projection.project_geometry(
                Point(origin_x, origin_y), crs=self.crs, to_crs="epsg:4326"
            )[0].coords[0]
            dest_lon, dest_lat = ox.projection.project_geometry(
                Point(dest_x, dest_y), crs=self.crs, to_crs="epsg:4326"
            )[0].coords[0]
            origin_h3 = h3.latlng_to_cell(origin_lat, origin_lon, res=self.h3_resolution)
            dest_h3 = h3.latlng_to_cell(dest_lat, dest_lon, res=self.h3_resolution)

            new_demand_units.append(
                DemandUnit(
                    demand_id=self.counter,
                    origin_node=origin,
                    destination_node=destination,
                    origin_lat=origin_lat,
                    origin_lon=origin_lon,
                    destination_lat=dest_lat,
                    destination_lon=dest_lon,
                    origin_h3_cell=origin_h3,
                    destination_h3_cell=dest_h3,
                )
            )
            self.counter += 1

        print(f"✅ Generated {len(new_demand_units)} new demand units")
        self.demand_units.extend(new_demand_units)

    def get_active_demands(self):
        return [d for d in self.demand_units if d.active]

    def step(self):
        for demand in self.demand_units:
            if demand.active:
                demand.step()