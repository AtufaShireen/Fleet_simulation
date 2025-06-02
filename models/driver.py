# models/driver.py
import random
import networkx as nx
import osmnx as ox
import h3
import logging
from shapely.geometry import Point
from typing import List, Tuple
from models.demand import DemandUnit

logging.basicConfig(level=logging.INFO)

class Driver:
    def __init__(self, driver_id, initial_location, vehicle_id, lat, lon, h3_cell, h3_resolution, config: dict = None):
        self.driver_id = driver_id
        self.location = initial_location  # (x, y) in projected CRS
        self.vehicle_id = vehicle_id
        self.status = "idle"
        self.path = []
        self.current_node = None
        self.insight_node = None
        self.current_demand = None
        self.insight_propensity = config.get("driver", {}).get("insight_propensity", 0.8)
        self.revenue = 0
        self.lat = lat
        self.lon = lon
        self.h3_cell = h3_cell
        self.h3_resolution = h3_resolution
        self.total_trips = 0
        self.total_idle_time = 0
        self.insight_acceptances = 0
        self.insight_attempts = 0
        self.path_index: int = 0
        self.home_location = initial_location
        self.home_lat = lat
        self.home_lon = lon
        self.home_h3_cell = h3_cell
        self.home_node = None

    def is_available(self):
        return self.status == "idle"
    
    def update_h3_cell(self):
        try:
            if not (17.0 <= self.lat <= 18.0 and 78.0 <= self.lon <= 78.7):
                logging.warning(f"Driver {self.driver_id} invalid lat/lon: ({self.lat}, {self.lon})")
            self.h3_cell = h3.latlng_to_cell(self.lat, self.lon, self.h3_resolution)
            if not h3.is_valid_cell(self.h3_cell):
                logging.error(f"Driver {self.driver_id} invalid H3 cell: {self.h3_cell}")
        except Exception as e:
            logging.error(f"Error updating H3 cell for driver {self.driver_id}: {e}")

    def update_node(self, G: nx.MultiDiGraph):
        try:
            self.current_node = ox.distance.nearest_nodes(G, self.location[0], self.location[1])
            x, y = G.nodes[self.current_node]["x"], G.nodes[self.current_node]["y"]
            crs = G.graph["crs"]
            point = ox.projection.project_geometry(Point(x, y), crs=crs, to_crs="epsg:4326")[0]
            lon, lat = point.coords[0]
            self.lat, self.lon = lat, lon
            logging.debug(f"Driver {self.driver_id} updated node: node={self.current_node}, lat={self.lat}, lon={self.lon}")
            self.update_h3_cell()
            if self.home_location == self.location and self.home_node is None:
                self.home_node = self.current_node
        except Exception as e:
            logging.error(f"Error updating node for driver {self.driver_id}: {e}")

    def receive_insight(self, node_id: int, G: nx.MultiDiGraph):
        self.insight_node = node_id
        try:
            path = nx.shortest_path(G, self.current_node, node_id, weight="length")
            coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in path]
            self.set_path(coords)
            self.status = "repositioning"
            logging.info(f"Driver {self.driver_id} received insight to node {node_id}")
        except nx.NetworkXNoPath:
            self.status = "idle"
            self.insight_node = None
            logging.warning(f"Driver {self.driver_id} failed to find path to insight node {node_id}")

    def set_path(self, path_coords: List[Tuple[float, float]]):
        self.path = path_coords
        self.path_index = 0

    def start_trip(self, demand: DemandUnit, G: nx.MultiDiGraph, fare: float = None):
        try:
            self.current_demand = demand
            self.status = "on_trip"
            self.total_trips += 1
            if fare is not None:
                self.revenue += fare
            elif self.path:
                path_nodes = [self.current_node]
                for x, y in self.path:
                    node = ox.distance.nearest_nodes(G, x, y)
                    if not path_nodes or node != path_nodes[-1]:
                        path_nodes.append(node)
                try:
                    distance = nx.path_weight(G, path_nodes, weight="length")
                    self.revenue += distance * 0.5
                except Exception as e:
                    logging.error(f"Driver {self.driver_id} failed to compute revenue: {e}")
            logging.debug(f"Driver {self.driver_id} started trip for demand {demand.demand_id}, revenue={self.revenue}")
        except Exception as e:
            logging.error(f"Driver {self.driver_id} failed to start trip: {e}")
            self.status = "idle"
            self.current_demand = None

    def complete_trip(self, demand: DemandUnit, graph: nx.MultiDiGraph):
        try:
            self.current_node = demand.destination_node
            self.location = (graph.nodes[self.current_node]["x"], graph.nodes[self.current_node]["y"])
            crs = graph.graph["crs"]
            point = ox.projection.project_geometry(Point(*self.location), crs=crs, to_crs="epsg:4326")[0]
            lon, lat = point.coords[0]
            self.lat, self.lon = lat, lon
            self.h3_cell = h3.latlng_to_cell(self.lat, self.lon, self.h3_resolution)
            self.status = "idle"
            self.current_demand = None
            self.set_path([])
            self.total_idle_time = 0
            logging.info(f"Driver {self.driver_id} completed trip for demand {demand.demand_id} at node {self.current_node}")
        except Exception as e:
            logging.error(f"Driver {self.driver_id} failed to complete trip: {e}")
            self.status = "idle"
            self.current_demand = None
            self.set_path([])

    def step_along_path(self, G: nx.MultiDiGraph, steps: int = 25):
        if not self.path or self.path_index >= len(self.path):
            if self.status == "on_trip" and self.current_demand:
                self.complete_trip(self.current_demand, G)
            elif self.status == "repositioning":
                self.status = "idle"
            self.path = []
            self.path_index = 0
            return

        new_index = min(self.path_index + steps, len(self.path))
        self.path_index = new_index
        self.location = self.path[self.path_index - 1] if self.path_index > 0 else self.path[0]
        crs = G.graph["crs"]
        point = ox.projection.project_geometry(Point(*self.location), crs=crs, to_crs="epsg:4326")[0]
        lon, lat = point.coords[0]
        self.lat, self.lon = lat, lon
        self.h3_cell = h3.latlng_to_cell(self.lat, self.lon, self.h3_resolution)
        self.current_node = ox.distance.nearest_nodes(G, self.location[0], self.location[1])

        if self.path_index >= len(self.path):
            if self.status == "on_trip" and self.current_demand:
                self.complete_trip(self.current_demand, G)
            elif self.status == "repositioning":
                self.status = "idle"
            self.path = []
            self.path_index = 0
            logging.info(f"Driver {self.driver_id} finished path: status={self.status}")

    def wait(self, G: nx.MultiDiGraph):
        self.total_idle_time += 1
        self.update_node(G)

    def decide_action(self, graph: nx.MultiDiGraph, insight_model=None):
        if self.status != "idle":
            logging.debug(f"Driver {self.driver_id} is {self.status}, skipping insight check")
            return
        if insight_model:
            insight_node = insight_model.get_insight_for_driver(self.driver_id)
            if insight_node:
                self.insight_attempts += 1
                if random.random() < self.insight_propensity:
                    self.receive_insight(insight_node, graph)
                    self.insight_acceptances += 1
                else:
                    logging.debug(f"Driver {self.driver_id} ignored insight due to propensity")
                    self.wait(graph)
            else:
                logging.debug(f"Driver {self.driver_id} has no insight")
                self.wait(graph)
        else:
            logging.debug(f"Driver {self.driver_id} has no insight model")
            self.wait(graph)