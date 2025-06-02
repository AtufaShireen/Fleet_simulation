# models/insight.py
from typing import List
import random
from models.driver import Driver
from models.demand import DemandUnit
import networkx as nx
import osmnx as ox
import h3
from utils.optimization_manager import OptimizationManager
import logging
from collections import defaultdict
from scipy.spatial import KDTree
import numpy as np

logging.basicConfig(level=logging.INFO)

class InsightModel:
    def __init__(self, demand: List[DemandUnit], seed: int = 42):
        self.demand = demand
        self.insights = {}  # driver_id -> suggested node_id
        self.random = random.Random(seed)

    def generate_insights(self, drivers: List[Driver], G: nx.MultiDiGraph):
        for driver in drivers:
            if not driver.is_available() or driver.path:
                continue
            if random.random() < driver.insight_propensity:
                try:
                    if driver.current_node is None:
                        driver.update_node(G)
                    current_node = driver.current_node
                    nearby_nodes = nx.single_source_dijkstra_path_length(G, current_node, cutoff=100, weight="length")
                    candidates = [n for n, dist in nearby_nodes.items() if dist > 10]
                    if not candidates:
                        continue
                    selected_node = random.choice(candidates)
                    driver.receive_insight(selected_node, G)
                except Exception as e:
                    logging.warning(f"[InsightModel] Error for Driver {driver.driver_id}: {e}")

    def generate_offline_insights(self, drivers: List[Driver], G: nx.MultiDiGraph, config: dict = None):
        logging.info("🚦 Starting offline optimization...")
        config = config or {}
        h3_resolution = config.get("h3_resolution", 9)

        # Build demand_counts and H3-to-node mapping
        demand_counts = {}
        h3_to_nodes = defaultdict(list)
        for i, demand_unit in enumerate(self.demand):
            try:
                h3_cell = demand_unit.origin_h3_cell
                if h3_cell is None:
                    raise ValueError(f"Demand {i} is missing origin_h3_cell")
                demand_counts[h3_cell] = demand_counts.get(h3_cell, 0) + 1
                # Assume DemandUnit has origin_node; adjust if not
                if hasattr(demand_unit, 'origin_node') and demand_unit.origin_node is not None:
                    h3_to_nodes[h3_cell].append(demand_unit.origin_node)
                else:
                    logging.debug(f"Demand {i} has no origin_node, will use spatial index")
            except Exception as e:
                logging.warning(f"⚠️ Failed to process demand {i} (ID: {demand_unit.demand_id}): {e}")

        logging.info(f"✅ Created demand map with {len(demand_counts)} H3 cells")
        logging.info(f"H3 cells with nodes: {len(h3_to_nodes)}")

        # Fallback: Build spatial index if no origin_node
        if not h3_to_nodes:
            logging.info("No origin_node in DemandUnit, building spatial index")
            node_coords = []
            node_ids = []
            for node in G.nodes:
                try:
                    lat = G.nodes[node]['y']
                    lon = G.nodes[node]['x']
                    node_coords.append((lat, lon))
                    node_ids.append(node)
                except KeyError:
                    continue
            if node_coords:
                kdtree = KDTree(node_coords)
                for h3_cell in demand_counts:
                    lat, lon = h3.cell_to_latlng(h3_cell)
                    _, idx = kdtree.query([lat, lon], k=1)
                    h3_to_nodes[h3_cell].append(node_ids[idx])
            else:
                logging.error("Failed to build spatial index: no valid node coordinates")

        optimizer = OptimizationManager(drivers, demand_counts, h3_resolution=h3_resolution, config=config)
        result = optimizer.run_optimization(use_heuristic=False)

        assignments = result["assignments"]
        logging.info(f"ILP Assignments: {assignments}")
        zone_indices = [zone_idx for zone_idx, _ in assignments]
        unique_zones = len(set(zone_indices))
        logging.info(f"Unique zone indices assigned: {unique_zones}")

        assigned_nodes = set()
        for zone_idx, driver_idx in assignments:
            zone_id = list(demand_counts.keys())[zone_idx]
            logging.info(f"Assigning driver {driver_idx} to zone {zone_idx} (H3={zone_id})")
            node = None
            # Try hotspot nodes first
            if zone_id in h3_to_nodes and h3_to_nodes[zone_id]:
                candidates = [n for n in h3_to_nodes[zone_id] if n not in assigned_nodes]
                if candidates:
                    node = random.choice(candidates)
                    logging.info(f"Selected hotspot node {node} for H3 {zone_id}")
            # Fallback: Spatial index or nearby nodes
            if node is None:
                try:
                    lat, lon = h3.cell_to_latlng(zone_id)
                    nodes, dists = ox.distance.nearest_nodes(G, lon, lat, k=5, return_dist=True)
                    for candidate_node, dist in zip(nodes, dists):
                        if candidate_node not in assigned_nodes and dist < 1000:
                            node = candidate_node
                            break
                    if node is None:
                        # Last resort: Random nearby node from driver’s position
                        nearby_nodes = nx.single_source_dijkstra_path_length(G, drivers[driver_idx].current_node, cutoff=1000, weight="length")
                        candidates = [n for n, dist in nearby_nodes.items() if n not in assigned_nodes and dist > 10]
                        node = random.choice(candidates) if candidates else None
                except Exception as e:
                    logging.warning(f"Failed to find node for H3 {zone_id}: {e}")
            if node is None:
                logging.error(f"No unique node found for driver {driver_idx}, zone {zone_idx}")
                continue
            assigned_nodes.add(node)
            driver = drivers[driver_idx]
            driver.receive_insight(node, G)
            self.insights[driver.driver_id] = node
            logging.info(f"Driver {driver_idx} assigned to node {node}")

        logging.info(f"🎯 Generated {len(self.insights)} offline insights")
        logging.info(f"🗺 Assignments: {self.insights}")

    def get_insight_for_driver(self, driver_id):
        return self.insights.get(driver_id, None)