# simulation/scheduler.py
import logging
from models.driver import Driver
from models.insight import InsightModel
from models.demand import DemandModel
from models.trip import TripModel
import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(message)s")

class Scheduler:
    def __init__(self, insight_interval: int, demand_rate: int, drivers: list[Driver], demands: list):
        self.insight_interval = insight_interval
        self.demand_rate = demand_rate
        self.drivers = drivers
        self.demands = demands
        if insight_interval <= 0:
            raise ValueError("insight_interval must be positive")
        if demand_rate < 0:
            raise ValueError("demand_rate cannot be negative")
        if not drivers:
            raise ValueError("drivers list cannot be empty")

    def step(
        self,
        step: int,
        drivers: list[Driver],
        demand_model: DemandModel,
        insight_model: InsightModel,
        trip_model: TripModel,
        G: nx.MultiDiGraph,
        config: dict = None
    ):
        config = config or {}
        max_trip_distance = config.get("max_trip_distance", 5000)

        # Step 1: Generate new demand units
        if step % self.insight_interval == 0 or step == 0:
            logging.info(f"Scheduler: Generating {self.demand_rate} new demand units at step {step}")
            demand_model.generate_demand(
                G=G,
                num_units=self.demand_rate,
                max_distance_m=max_trip_distance
            )

        # Step 2: Generate insights every `insight_interval` steps
        if step % self.insight_interval == 0:
            logging.info(f"Scheduler: Generating insights at step {step}")
            insight_model.generate_insights(drivers, G)

        # Step 3: Each driver decides what to do
        for driver in drivers:
            driver.decide_action(graph=G, insight_model=insight_model)

        # Step 4: Assign trips to drivers
        logging.info(f"Scheduler: Assigning trips to drivers at step {step}")
        active_demands = demand_model.get_active_demands()
        trip_model.assign_trips(G, drivers, active_demands)
        logging.info(f"Scheduler: Assigned trips to {sum(1 for d in drivers if d.status == 'on_trip')} drivers")

        # Step 5: Move drivers and update demand lifecycle
        logging.info("Scheduler: Moving drivers and updating demand lifecycle")
        for driver in drivers:
            if driver.status in ("on_trip", "repositioning"):
                driver.step_along_path(G)
            else:
                driver.wait(G)

        # Step 6: Advance all active demand timers
        demand_model.step()