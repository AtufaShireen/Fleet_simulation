# simulation/engine.py
import logging
from collections import defaultdict
from utils.visualization import animate_from_log
import os
from pyproj import Transformer

logging.basicConfig(level=logging.INFO, format="%(message)s")

class SimulationEngine:
    def __init__(self, config, drivers, demand_model, insight_model, trip_model, scheduler, road_graph=None):
        self.config = config
        self.drivers = drivers
        self.demand_model = demand_model
        self.insight_model = insight_model
        self.trip_model = trip_model
        self.scheduler = scheduler
        self.road_graph = road_graph
        self.driver_positions_log = []
        self.metrics = defaultdict(list)
        self.transformer = Transformer.from_crs("EPSG:32643", "EPSG:4326")

        required_keys = ["steps_per_epoch", "h3_resolution", "max_trip_distance"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

    def run_epoch(self, epoch_num=0):
        logging.info(f"🌅 Starting new day (Epoch {epoch_num + 1})")
        self.driver_positions_log = []

        logging.info("📍 Driver starting positions:")
        for driver in self.drivers:
            logging.info(f"Driver {driver.driver_id}: latlon={driver.lat, driver.lon}, location={driver.location}, h3_cell={driver.h3_cell}, node={driver.current_node}")

        for driver in self.drivers:
            driver.status = "idle"
            driver.path = []
            driver.insight_node = None
            driver.revenue = 0
            driver.total_trips = 0
            driver.total_idle_time = 0
            driver.insight_acceptances = 0
            driver.insight_attempts = 0
            driver.location = driver.home_location
            driver.lat = driver.home_lat
            driver.lon = driver.home_lon
            driver.h3_cell = driver.home_h3_cell
            driver.current_node = driver.home_node
            driver.update_node(self.road_graph)
        logging.info(f"🚗 Reset {len(self.drivers)} drivers to home locations")

        self.trip_model.reset_cache()

        logging.info("📍 Driver positions after home reset:")
        for driver in self.drivers:
            logging.info(f"Driver {driver.driver_id}: latlon={driver.lat, driver.lon}, location={driver.location}, h3_cell={driver.h3_cell}, node={driver.current_node}")

        logging.info("🧠 Generating offline insights for new day...")
        self.insight_model.generate_offline_insights(self.drivers, self.road_graph, self.config)

        for step in range(self.config["steps_per_epoch"]):
            self.scheduler.step(
                step,
                self.drivers,
                self.demand_model,
                self.insight_model,
                self.trip_model,
                self.road_graph,
                config=self.config
            )
            self.log_step(step)

        logging.info("📍 Driver ending positions:")
        for driver in self.drivers:
            logging.info(f"Driver {driver.driver_id}: latlon={driver.lat, driver.lon}, location={driver.location}, h3_cell={driver.h3_cell}, node={driver.current_node}")

        viz_config = self.config.get("visualization", {})
        if self.road_graph and (viz_config.get("animate", False) or viz_config.get("save", False)):
            if not self.road_graph.nodes:
                logging.info("Empty road graph, skipping visualization")
                return
            filename = viz_config.get("filename", "driver_animation.mp4")
            name, ext = os.path.splitext(filename)
            epoch_filename = f"{name}_epoch{epoch_num}{ext}"
            logging.info(f"🎥 Generating visualization for epoch {epoch_num}")
            animate_from_log(
                self.road_graph,
                self.driver_positions_log,
                interval=300,
                save_path=epoch_filename if viz_config.get("save", False) else None,
                show=viz_config.get("animate", False)
            )

    def log_step(self, step):
        step_positions = []
        for driver in self.drivers:
            try:
                lat, lon = driver.lat, driver.lon
                x,y = driver.location
                if not (206952 <= x <= 247270 and 1914151 <= y <= 1942803):
                    logging.info(f"Step {step}, Driver {driver.driver_id}: Invalid UTM=({x}, {y}), LatLon=({lat}, {lon})")
                    step_positions.append((driver.driver_id, (0, 0)))
                    continue
                # if lat is None or lon is None or not (78.3 <= lon <= 78.6 and 17.2 <= lat <= 17.5):
                #     logging.info(f"Step {step}, Driver {driver.driver_id}: Invalid LatLon=({lat}, {lon}), UTM={driver.location}")
                #     step_positions.append((driver.driver_id, (0, 0)))
                #     continue
                step_positions.append((driver.driver_id, (x, y)))
                # step_positions.append((driver.driver_id, (lon, lat)))
                logging.info(f"Step {step}, Driver {driver.driver_id}: LatLon=({lon}, {lat}), UTM={driver.location}, Node={driver.current_node}, Status={driver.status}")
            except Exception as e:
                logging.info(f"Step {step}, Driver {driver.driver_id}: Failed to process latlon: {e}")
                step_positions.append((driver.driver_id, (0, 0)))
        self.driver_positions_log.append(step_positions)
        logging.info(f"Step {step}: Position log entry: {step_positions}")

        active_demands = set(id(d) for d in self.demand_model.get_active_demands())
        fulfilled = 0
        unfulfilled = 0
        for d in self.demand_model.demand_units:
            if id(d) not in active_demands:
                if d.wait_time > 0:
                    fulfilled += 1
                else:
                    unfulfilled += 1

        total_revenue = sum(driver.revenue for driver in self.drivers)
        driver_busy = sum(1 for driver in self.drivers if driver.status == "on_trip")
        driver_util = driver_busy / len(self.drivers) if self.drivers else 0

        self.metrics["fulfilled_trips"].append(fulfilled)
        self.metrics["unfulfilled_demands"].append(unfulfilled)
        self.metrics["total_revenue"].append(total_revenue)
        self.metrics["driver_utilization"].append(driver_util)

        if step % 60 == 0:
            logging.info(f"[t={step}] Revenue: ₹{total_revenue:.1f} | Busy: {driver_busy}/{len(self.drivers)} | Fulfilled: {fulfilled} | Unfulfilled: {unfulfilled}")


    def summarize_results(self):
        print("\n📊 Final Simulation Summary:")
        for key, values in self.metrics.items():
            avg = sum(values) / len(values) if values else 0
            print(f"{key}: {avg:.2f}")
        avg_idle_time = sum(driver.total_idle_time for driver in self.drivers) / len(self.drivers) if self.drivers else 0
        avg_trips = sum(driver.total_trips for driver in self.drivers) / len(self.drivers) if self.drivers else 0
        insight_acceptances = sum(driver.insight_acceptances for driver in self.drivers)
        insight_attempts = sum(driver.insight_attempts for driver in self.drivers)
        insight_rate = insight_acceptances / insight_attempts if insight_attempts > 0 else 0
        print(f"average_driver_idle_time: {avg_idle_time:.2f}")
        print(f"average_driver_trips: {avg_trips:.2f}")
        print(f"insight_acceptance_rate: {insight_rate:.2f}")