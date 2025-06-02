# utils/optimization_manager.py
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus
import time
import math
from typing import List, Set, Tuple, Dict
import logging
import h3
from models.driver import Driver

logging.basicConfig(level=logging.INFO, format='%(message)s')

def h3_grid_distance(h3_cell1: str, h3_cell2: str, resolution: int) -> int:
    """Calculate H3 grid distance between two cells at the same resolution."""
    if not (h3.is_valid_cell(h3_cell1) and h3.is_valid_cell(h3_cell2)):
        logging.warning(f"Invalid H3 cell: cell1={h3_cell1}, cell2={h3_cell2}")
        return float('inf')
    if h3.get_resolution(h3_cell1) != resolution or h3.get_resolution(h3_cell2) != resolution:
        logging.warning(f"Resolution mismatch: cell1={h3_cell1} (res={h3.get_resolution(h3_cell1)}), cell2={h3_cell2} (res={h3.get_resolution(h3_cell2)}), expected={resolution}")
        return float('inf')
    try:
        dist = h3.grid_distance(h3_cell1, h3_cell2)
        return dist
    except (h3.H3ValueError, h3.H3FailedError) as e:
        logging.warning(f"H3 grid distance failed: {e} (cell1={h3_cell1}, cell2={h3_cell2})")
        return float('inf')

class DataLoader:
    def __init__(self, drivers: List[Driver], zone_demand: Dict[str, float], h3_resolution: int, config: dict = None):
        self.drivers = drivers
        self.driver_h3_cells = []
        self.h3_resolution = h3_resolution
        self.config = config or {}

        # Compute driver H3 cells
        for driver in drivers:
            try:
                if not (17.0 <= driver.lat <= 17.7 and 78.0 <= driver.lon <= 78.7):
                    raise ValueError(f"Invalid lat/lon for Hyderabad: ({driver.lat}, {driver.lon})")
                driver_h3 = h3.latlng_to_cell(driver.lat, driver.lon, self.h3_resolution)
                if not h3.is_valid_cell(driver_h3):
                    raise ValueError(f"Invalid H3 cell: {driver_h3}")
                self.driver_h3_cells.append(driver_h3)
            except Exception as e:
                logging.error(f"Invalid H3 cell for driver {driver.driver_id} at ({driver.lat}, {driver.lon}): {e}")
                self.driver_h3_cells.append(None)

        self.zone_demand = zone_demand
        self.zone_ids = list(zone_demand.keys())
        self.zone_index_map = {z: i for i, z in enumerate(self.zone_ids)}
        self.N = len(self.zone_ids)
        self.D = len(drivers)

        # Validate inputs
        if not drivers:
            raise ValueError("drivers list cannot be empty")
        if not zone_demand:
            raise ValueError("zone_demand cannot be empty")
        if all(h3_cell is None for h3_cell in self.driver_h3_cells):
            raise ValueError("All drivers have invalid h3_cell")
        for z in self.zone_ids:
            if not h3.is_valid_cell(z):
                raise ValueError(f"Invalid H3 cell in zone_demand: {z}")
            if h3.get_resolution(z) != self.h3_resolution:
                raise ValueError(f"Zone H3 mismatch: resolution {h3.get_resolution(z)} != {self.h3_resolution}")

        # Optimization parameters
        self.lambda_ = self.config.get("optimization", {}).get("lambda", 0.5)
        self.alpha = self.config.get("optimization", {}).get("alpha", 0.01)
        p_range = self.config.get("optimization", {}).get("p_range", [0.2, 1])
        s_range = self.config.get("optimization", {}).get("s_range", [1, 3])
        r_range = self.config.get("optimization", {}).get("r_range", [0, 10000])
        self.max_grid_distance = self.config.get("optimization", {}).get("max_grid_distance", 1)
        self.max_cells = self.config.get("optimization", {}).get("max_cells", 4)
        self.max_distance = self.config.get("optimization", {}).get("max_distance", 10000)

        self.v = np.array([self.zone_demand[z] for z in self.zone_ids])

        # Compute p (affinity): distance from driver home to zone
        self.p = np.zeros((self.N, self.D))
        for i, z in enumerate(self.zone_ids):
            for j, driver in enumerate(self.drivers):
                home_lat = getattr(driver, 'home_lat', driver.lat)
                home_lon = getattr(driver, 'home_lon', driver.lon)
                distance = self._compute_haversine_distance(home_lat, home_lon, z)
                p = 1 - min(distance, 7000) / 7000
                self.p[i, j] = np.clip(p, 0.2, 1)

        # Compute s (surge): logarithmic demand-to-driver ratio
        driver_counts = np.zeros(self.N)
        for j, h3_cell in enumerate(self.driver_h3_cells):
            if h3_cell in self.zone_ids:
                i = self.zone_index_map[h3_cell]
                driver_counts[i] += 1
        self.s = np.zeros(self.N)
        for i in range(self.N):
            demand_count = self.v[i]
            driver_count = driver_counts[i] if driver_counts[i] > 0 else 1
            s = 1 + 2 * np.log(1 + demand_count / driver_count)
            self.s[i] = np.clip(s, 1, 3)

        # Compute r (revenue) and offline priority
        self.r = np.array([driver.revenue for driver in drivers])
        self.r_priority = np.array([1 / (1 + self.alpha * self.r[j]) for j in range(self.D)])

        self.F = self._generate_feasible_pairs()

    def _compute_haversine_distance(self, lat1: float, lon1: float, zone_id: str) -> float:
        """Compute Haversine distance (meters) from lat/lon to zone centroid."""
        try:
            zone_lat, zone_lon = h3.cell_to_latlng(zone_id)
            R = 6371000  # Earth radius in meters
            phi1, phi2 = math.radians(lat1), math.radians(zone_lat)
            delta_phi = math.radians(zone_lat - lat1)
            delta_lambda = math.radians(zone_lon - lon1)
            a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            return R * c
        except Exception as e:
            logging.warning(f"Haversine distance failed for ({lat1}, {lon1}), zone {zone_id}: {e}")
            return float("inf")

    def _generate_feasible_pairs(self) -> Set[Tuple[int, int]]:
        feasible_pairs = set()
        unmatched_drivers = []

        for j, (driver, driver_h3) in enumerate(zip(self.drivers, self.driver_h3_cells)):
            if not h3.is_valid_cell(driver_h3):
                logging.error(f"Driver {j} has invalid H3 cell: {driver_h3}")
                unmatched_drivers.append((j, driver_h3))
                continue
            nearby_h3 = h3.grid_disk(driver_h3, self.max_grid_distance)
            logging.info(f"Driver {j}: H3={driver_h3}, grid_disk returned {len(nearby_h3)} cells")
            zone_distances = []
            for z in nearby_h3:
                if h3.get_resolution(z) != self.h3_resolution:
                    logging.warning(f"grid_disk cell {z} resolution {h3.get_resolution(z)} != {self.h3_resolution}")
                    continue
                if z in self.zone_index_map:
                    i = self.zone_index_map[z]
                    distance = self._compute_haversine_distance(driver.lat, driver.lon, z)
                    if distance <= self.max_distance:
                        demand = self.zone_demand.get(z, 0)
                        zone_distances.append((i, z, distance, demand))
            zone_distances.sort(key=lambda x: x[3], reverse=True)
            selected_zones = zone_distances[:self.max_cells]
            matched = len(selected_zones)
            logging.info(f"Driver {j}: H3={driver_h3}, Matches={matched}")
            if len(selected_zones) < self.max_cells:
                for z in self.zone_ids:
                    if z not in nearby_h3:
                        i = self.zone_index_map[z]
                        distance = self._compute_haversine_distance(driver.lat, driver.lon, z)
                        if distance <= self.max_distance:
                            demand = self.zone_demand.get(z, 0)
                            zone_distances.append((i, z, distance, demand))
                zone_distances.sort(key=lambda x: x[3], reverse=True)
                selected_zones = zone_distances[:self.max_cells]
                matched = len(selected_zones)
                logging.info(f"Driver {j}: Expanded to <{self.max_distance}m, Selected={matched}")
            for i, _, _, _ in selected_zones:
                feasible_pairs.add((i, j))
            if matched == 0:
                unmatched_drivers.append((j, driver_h3))

        for j, driver_h3 in unmatched_drivers:
            if not h3.is_valid_cell(driver_h3):
                logging.error(f"Skipping fallback for driver {j} due to invalid H3 cell: {driver_h3}")
                continue
            zone_distances = []
            for z in self.zone_ids:
                distance = self._compute_haversine_distance(self.drivers[j].lat, self.drivers[j].lon, z)
                if distance <= self.max_distance:
                    i = self.zone_index_map[z]
                    demand = self.zone_demand.get(z, 0)
                    zone_distances.append((i, z, distance, demand))
            if zone_distances:
                zone_distances.sort(key=lambda x: x[3], reverse=True)
                selected_zones = zone_distances[:self.max_cells]
                for i, _, _, _ in selected_zones:
                    feasible_pairs.add((i, j))
                logging.info(f"Assigned Driver {j} (H3={driver_h3}) to {len(selected_zones)} zones via distance-based fallback")
            else:
                logging.error(f"Failed to assign Driver {j} (H3={driver_h3}): No zones within {self.max_distance}m")

        logging.info(f"Generated {len(feasible_pairs)} feasible pairs (grid_disk + distance-based filtering, with {len(unmatched_drivers)} fallbacks)")
        return feasible_pairs

    def get_distance_weight(self, i: int, j: int) -> float:
        z_h3 = self.zone_ids[i]
        d_h3 = self.driver_h3_cells[j]
        grid_dist = h3_grid_distance(z_h3, d_h3, self.h3_resolution)
        max_trip_distance = self.config.get("max_trip_distance", 5000) / 1000
        if grid_dist == 0:
            return 1.0
        elif grid_dist <= 1:
            return 0.8
        elif grid_dist <= 2:
            return 0.6
        elif grid_dist <= 3:
            return 0.4
        else:
            return 0.0

    def update_dynamic_data(self):
        # Update p, s, r, r_priority
        for i, z in enumerate(self.zone_ids):
            for j, driver in enumerate(self.drivers):
                home_lat = getattr(driver, 'home_lat', driver.lat)
                home_lon = getattr(driver, 'home_lon', driver.lon)
                distance = self._compute_haversine_distance(home_lat, home_lon, z)
                p = 1 - min(distance, 7000) / 7000
                self.p[i, j] = np.clip(p, 0.2, 1)

        driver_counts = np.zeros(self.N)
        for j, h3_cell in enumerate(self.driver_h3_cells):
            if h3_cell in self.zone_ids:
                i = self.zone_index_map[h3_cell]
                driver_counts[i] += 1
        for i in range(self.N):
            demand_count = self.v[i]
            driver_count = driver_counts[i] if driver_counts[i] > 0 else 1
            s = 1 + 2 * np.log(1 + demand_count / driver_count)
            self.s[i] = np.clip(s, 1, 3)

        self.r = np.array([driver.revenue for driver in self.drivers])
        self.r_priority = np.array([1 / (1 + self.alpha * self.r[j]) for j in range(self.D)])
        self.F = self._generate_feasible_pairs()
        logging.info("Dynamic H3 data updated (affinity, surge, revenue, priority, F).")

class DynamicMCalculator:
    def __init__(self, config: dict = None):
        config = config or {}
        opt_config = config.get("optimization", {})
        self.M_min = opt_config.get("M_min", 8)
        self.beta = opt_config.get("beta", 0.8)
        self.gamma = opt_config.get("gamma", 0.7)

    def calculate_M(self, D: int, s: np.ndarray, N: int) -> int:
        total_surge = np.sum(s)
        surge_excess = (total_surge - N) / total_surge if total_surge > 0 else 0
        M_t = max(self.M_min, int(D * (self.beta - self.gamma * surge_excess)))
        M_t = min(M_t, D)
        logging.info(f"Calculated M_t = {M_t} with total surge = {total_surge:.2f}")
        return M_t

class ILPSolver:
    def __init__(self, data: DataLoader, M_t: int):
        self.data = data
        self.M_t = M_t
        self.model = None
        self.x = None
        self.y = None

    def formulate(self):
        self.model = LpProblem("Driver_Zone_Allocation", LpMaximize)
        self.x = {(i, j): LpVariable(f"x_{i}_{j}", cat="Binary") for (i, j) in self.data.F}
        self.y = {i: LpVariable(f"y_{i}", cat="Binary") for i in range(self.data.N)}
        offline = lpSum(self.data.v[i] * self.data.get_distance_weight(i, j) * self.data.p[i, j] * self.x[i, j]
                       for (i, j) in self.data.F)
        online = lpSum((self.data.s[i] * self.data.v[i]) * self.data.get_distance_weight(i, j) *
                      self.data.r_priority[j] * self.x[i, j]
                      for (i, j) in self.data.F)
        self.model += self.data.lambda_ * offline + (1 - self.data.lambda_) * online
        for j in range(self.data.D):
            self.model += lpSum(self.x[i, j] for i, jj in self.data.F if jj == j) == 1, f"Driver_{j}_one_zone"
        for i in range(self.data.N):
            self.model += lpSum(self.x[i, j] for ii, j in self.data.F if ii == i) <= 1, f"Zone_{i}_at_most_one_driver"
        self.model += lpSum(self.y[i] for i in range(self.data.N)) >= self.M_t, "Min_zone_coverage"
        for i in range(self.data.N):
            for j in [jj for ii, jj in self.data.F if ii == i]:
                self.model += self.y[i] >= self.x[i, j], f"Link_y_{i}_x_{i}_{j}"
            self.model += self.y[i] <= lpSum(self.x[i, j] for ii, j in self.data.F if ii == i), f"Link_y_{i}_sum_x_{i}"

    def solve(self) -> Dict:
        start_time = time.time()
        self.model.solve()
        status = LpStatus[self.model.status]
        result = {
            "status": status,
            "objective": self.model.objective.value() if status == "Optimal" else None,
            "assignments": [(i, j) for (i, j) in self.data.F if self.x[i, j].value() > 0.5] if status == "Optimal" else [],
            "covered_zones": [i for i in range(self.data.N) if self.y[i].value() > 0.5] if status == "Optimal" else [],
            "solve_time": time.time() - start_time
        }
        logging.info(f"ILP Status: {status}, Solve Time: {result['solve_time']:.3f}s, Objective: {result['objective']}")
        return result

class SimulatedAnnealingSolver:
    def __init__(self, data: DataLoader, M_t: int, initial_temp: float = 1000, cooling_rate: float = 0.995):
        self.data = data
        self.M_t = M_t
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def objective_value(self, assignments: List[Tuple[int, int]]) -> float:
        offline = sum(self.data.v[i] * self.data.get_distance_weight(i, j) * self.data.p[i, j]
                     for i, j in assignments)
        online = sum((self.data.s[i] * self.data.v[i]) * self.data.get_distance_weight(i, j) *
                    self.data.r_priority[j]
                    for i, j in assignments)
        return self.data.lambda_ * offline + (1 - self.data.lambda_) * online

    def is_valid(self, assignments: List[Tuple[int, int]]) -> bool:
        driver_count = {}
        zone_count = {}
        for i, j in assignments:
            driver_count[j] = driver_count.get(j, 0) + 1
            zone_count[i] = zone_count.get(i, 0) + 1
            if driver_count[j] > 1 or zone_count[i] > 1:
                return False
        if len(driver_count) != self.data.D:
            return False
        covered_zones = len(set(i for i, _ in assignments))
        if covered_zones < self.M_t:
            return False
        return all((i, j) in self.data.F for i, j in assignments)

    def generate_initial_solution(self) -> List[Tuple[int, int]]:
        assignments = []
        available_drivers = list(range(self.data.D))
        available_zones = list(range(self.data.N))
        np.random.shuffle(available_drivers)
        for j in available_drivers:
            feasible_zones = [i for i in available_zones if (i, j) in self.data.F]
            if feasible_zones:
                i = np.random.choice(feasible_zones)
                assignments.append((i, j))
                available_zones.remove(i)
        return assignments if self.is_valid(assignments) else self.generate_initial_solution()

    def get_neighbor(self, assignments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        new_assignments = assignments.copy()
        idx1, idx2 = np.random.choice(len(assignments), 2, replace=False)
        i1, j1 = new_assignments[idx1]
        i2, j2 = new_assignments[idx2]
        new_assignments[idx1] = (i1, j2)
        new_assignments[idx2] = (i2, j1)
        if self.is_valid(new_assignments):
            return new_assignments
        idx = np.random.choice(len(assignments))
        i, j = new_assignments[idx]
        feasible_zones = [z for z in range(self.data.N) if (z, j) in self.data.F and z not in [ii for ii, _ in new_assignments]]
        if feasible_zones:
            new_i = np.random.choice(feasible_zones)
            new_assignments[idx] = (new_i, j)
        return new_assignments if self.is_valid(new_assignments) else assignments

    def solve(self, max_iterations: int = 1000) -> Dict:
        start_time = time.time()
        current_solution = self.generate_initial_solution()
        current_value = self.objective_value(current_solution)
        best_solution = current_solution
        best_value = current_value
        temp = self.initial_temp
        for _ in range(max_iterations):
            neighbor = self.get_neighbor(current_solution)
            neighbor_value = self.objective_value(neighbor)
            if neighbor_value > current_value or np.random.random() < math.exp((neighbor_value - current_value) / temp):
                current_solution = neighbor
                current_value = neighbor_value
                if current_value > best_value:
                    best_solution = current_solution
                    best_value = current_value
            temp *= self.cooling_rate
        result = {
            "status": "Heuristic",
            "objective": best_value,
            "assignments": best_solution,
            "covered_zones": list(set(i for i, _ in best_solution)),
            "solve_time": time.time() - start_time
        }
        logging.info(f"SA Objective: {best_value}, Solve Time: {result['solve_time']:.3f}s, Covered Zones: {len(result['covered_zones'])}")
        return result

class OptimizationManager:
    def __init__(self, drivers: List[Driver], zone_demand: Dict[str, float], h3_resolution: int, config: dict = None):
        self.data_loader = DataLoader(drivers, zone_demand, h3_resolution, config)
        self.m_calculator = DynamicMCalculator(config)
        self.ilp_solver = None
        self.sa_solver = None

    def run_optimization(self, use_heuristic: bool = False) -> Dict:
        logging.info("Starting optimization cycle...")
        self.data_loader.update_dynamic_data()
        logging.info(f"Total drivers: {self.data_loader.D}")
        logging.info(f"Total zones: {self.data_loader.N}")
        M_t = self.m_calculator.calculate_M(
            self.data_loader.D, self.data_loader.s, self.data_loader.N
        )
        logging.info(f"Calculated M_t: {M_t}")
        feasible_zones = len(set(i for i, _ in self.data_loader.F))
        logging.info(f"Feasible zones from F: {feasible_zones}")
        logging.info(f"Total feasible pairs: {len(self.data_loader.F)}")
        if M_t > self.data_loader.D or M_t > feasible_zones:
            logging.error(f"Infeasible: M_t={M_t}, D={self.data_loader.D}, Feasible Zones={feasible_zones}")
            logging.info(f"Feasible zones list: {list(set(i for i, _ in self.data_loader.F))}")
            return {"status": "Infeasible", "objective": None, "assignments": [], "covered_zones": []}
        if use_heuristic:
            self.sa_solver = SimulatedAnnealingSolver(self.data_loader, M_t)
            result = self.sa_solver.solve()
        else:
            self.ilp_solver = ILPSolver(self.data_loader, M_t)
            self.ilp_solver.formulate()
            result = self.ilp_solver.solve()
        return result
    

