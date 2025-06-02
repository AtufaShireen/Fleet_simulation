# models/trip.py
from typing import List, Tuple
from models.driver import Driver
from models.demand import DemandUnit
import networkx as nx
import logging
import h3

logging.basicConfig(level=logging.INFO, format="%(message)s")

def assign_driver_trips(driver: Driver, demands: List[DemandUnit], G: nx.MultiDiGraph, max_distance: float, fare_rate: float, distance_cache: dict, h3_resolution: int) -> Tuple[int, List, float, DemandUnit]:
    """Assign the first demand in the driver's current H3 zone, sorted by wait_time."""
    if not driver.is_available():
        logging.debug(f"Driver {driver.driver_id} is not available, skipping trip assignment")
        return driver.driver_id, [], 0, None

    try:
        # Use driver's current H3 cell
        # current_zone_h3 = h3.latlng_to_cell(driver.lat, driver.lon, h3_resolution)
        current_zone_h3 = driver.h3_cell
    except Exception as e:
        logging.warning(f"Driver {driver.driver_id} failed to determine H3 zone: {e}")
        return driver.driver_id, [], 0, None

    # Filter and sort demands by wait_time
    zone_demands = [
        d for d in demands
        if d.active and d.origin_h3_cell == current_zone_h3
    ]
    if not zone_demands:
        logging.debug(f"Driver {driver.driver_id} found no active demands in zone {current_zone_h3}")
        return driver.driver_id, [], 0, None

    selected_demand = max(zone_demands, key=lambda d: d.wait_time)

    # Compute trip path
    try:
        if driver.current_node is None:
            driver.update_node(G)
        path_to_origin = nx.shortest_path(G, driver.current_node, selected_demand.origin_node, weight="length")
        path_to_destination = nx.shortest_path(G, selected_demand.origin_node, selected_demand.destination_node, weight="length")
        full_path = path_to_origin + path_to_destination[1:]
        total_distance = nx.path_weight(G, full_path, weight="length")
        
        # Check max_distance
        if total_distance > max_distance:
            logging.debug(f"Driver {driver.driver_id} rejected demand {selected_demand.demand_id}: distance {total_distance}m > {max_distance}m")
            return driver.driver_id, [], 0, None

        coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in full_path]
        fare = total_distance * fare_rate
        logging.info(f"✅ Assigned: Driver {driver.driver_id} | Demand {selected_demand.demand_id} | Distance: {int(total_distance)}m | Steps: {len(coords)} | Wait Time: {selected_demand.wait_time}")
        return driver.driver_id, coords, fare, selected_demand
    except nx.NetworkXNoPath:
        logging.warning(f"Driver {driver.driver_id} failed to compute path for demand {selected_demand.demand_id}")
        return driver.driver_id, [], 0, None

class TripModel:
    def __init__(self, max_distance: int = 5000, fare_rate: float = 0.5):
        self.max_distance = max_distance
        self.fare_rate = fare_rate
        self.distance_cache = {}

    def reset_cache(self):
        """Clear the distance cache at the start of each epoch."""
        self.distance_cache = {}
        logging.info("Cleared trip distance cache")

    def assign_trips(self, G: nx.MultiDiGraph, drivers: List[Driver], demands: List[DemandUnit], h3_resolution: int = 7):
        """Assign trips to drivers sequentially."""
        logging.info(f"Assigning trips for {len(drivers)} drivers and {len([d for d in demands if d.active])} active demands")
        assignments = []
        for driver in drivers:
            try:
                driver_id, coords, fare, demand = assign_driver_trips(
                    driver, demands, G, self.max_distance, self.fare_rate, self.distance_cache, h3_resolution
                )
                if coords and demand:
                    assignments.append((driver_id, coords, fare, demand))
            except Exception as e:
                logging.error(f"Error assigning trip for driver {driver.driver_id}: {e}")

        driver_map = {d.driver_id: d for d in drivers}
        assigned_demands = set()
        for driver_id, coords, fare, demand in assignments:
            if demand in assigned_demands:
                continue
            driver = driver_map[driver_id]
            try:
                driver.set_path(coords)
                driver.start_trip(demand, G, fare=fare)
                demand.wait_time = 0  # Deactivate demand
                assigned_demands.add(demand)
            except Exception as e:
                logging.error(f"Error applying trip for driver {driver_id}: {e}")

        logging.info(f"Completed trip assignments: {len(assigned_demands)} trips assigned")