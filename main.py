# main.py
import yaml
import random
from simulation.engine import SimulationEngine
from simulation.scheduler import Scheduler
from utils.data_generation import load_city_graph, get_hotspot_nodes, initialize_drivers
from models.demand import DemandModel
from models.insight import InsightModel
from models.trip import TripModel


def main():
    # Load and validate config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    required_keys = [
        "city",
        "hotspots",
        "num_drivers",
        "num_demands",
        "epochs",
        "steps_per_epoch",
        "insight_interval",
        "demand_rate_per_step",
        "max_trip_distance",
        "h3_resolution",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    if config["h3_resolution"] not in [7, 8, 9, 10]:
        raise ValueError("h3_resolution must be between 7 and 10")
    if config["max_trip_distance"] <= 0:
        raise ValueError("max_trip_distance must be positive")

    # Set random seed for reproducibility
    random.seed(42)

    # Initialize graph and data
    print("🌐 Loading city graph and initializing data...")
    G = load_city_graph(config["city"])
    hotspots = get_hotspot_nodes(G, config["hotspots"], seed=42, h3_resolution=config["h3_resolution"])
    drivers = initialize_drivers(
        G, config["num_drivers"], hotspots, seed=42, h3_resolution=config["h3_resolution"]
    )

    # Initialize models
    print("🛠 Initializing models...")
    demand_model = DemandModel(
        hotspots, G, max_distance_m=config["max_trip_distance"], h3_resolution=config["h3_resolution"]
    )
    demand_model.generate_demand(G, num_units=config["num_demands"])
    demand_nodes = demand_model.get_active_demands()
    print(f"📦 Generated {len(demand_nodes)} active demand units")

    insight_model = InsightModel(demand_nodes)
    trip_model = TripModel(
        max_distance=config["max_trip_distance"],
        fare_rate=config.get("trip", {}).get("fare_rate", 0.5)
    )
    scheduler = Scheduler(
        insight_interval=config["insight_interval"],
        demand_rate=config["demand_rate_per_step"],
        drivers=drivers,
        demands=demand_model.demand_units,
    )

    sim = SimulationEngine(
        config=config,
        drivers=drivers,
        demand_model=demand_model,
        insight_model=insight_model,
        trip_model=trip_model,
        scheduler=scheduler,
        road_graph=G
    )

    # Run simulation
    for epoch in range(config["epochs"]):
        print(f"\n🚀 Running Epoch {epoch + 1}/{config['epochs']}")
        sim.run_epoch(epoch_num=epoch)

    print("\n📊 Generating summary:")
    sim.summarize_results()


if __name__ == "__main__":
    main()