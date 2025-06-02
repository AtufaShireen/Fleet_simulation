Fleet Simulation Documentation
Overview
This codebase implements a fleet simulation system designed to optimize driver allocation in a ride-hailing service using an insight model. The system leverages geospatial data, demand modeling, and optimization techniques to provide drivers with insights on high-demand zones, aiming to maximize revenue and driver utilization while ensuring coverage of demand zones. The simulation is built around a city’s road network, using Hyderabad, India, as the default city, and incorporates H3 geospatial indexing for efficient spatial computations.
Objectives

Driver Allocation Optimization: Assign drivers to high-demand zones to maximize trip fulfillment and revenue.
Demand Modeling: Simulate realistic passenger demand patterns based on hotspots and geospatial data.
Insight Generation: Provide drivers with actionable insights to reposition to high-demand areas, balancing immediate (online) and future (offline) demand.
Performance Metrics: Track and summarize key metrics such as fulfilled trips, unfulfilled demands, driver utilization, revenue, and insight acceptance rates.
Visualization: Optionally generate animations to visualize driver movements across the city.

Key Components

main.py: Entry point for the simulation, orchestrating initialization and execution.
config.yaml: Configuration file specifying simulation parameters.
models/: Contains classes for demand, driver, insight, and trip management.
simulation/: Manages the simulation engine and scheduling logic.
utils/: Utility functions for data generation, optimization, and visualization.

File-by-File Documentation
1. main.py
Purpose: Serves as the main entry point, initializing the simulation components and running the simulation loop.
Logic:

Loads and validates the configuration from config.yaml.
Initializes the city road graph, hotspot nodes, and drivers.
Sets up models for demand, insights, trips, and scheduling.
Runs the simulation for a specified number of epochs, invoking the simulation engine for each epoch.
Summarizes results after all epochs.

Optimizations:

Uses a fixed random seed (42) for reproducibility.
Validates configuration keys and values to prevent runtime errors.
Caches the city graph to avoid repeated downloads.

Key Functions:

main(): Orchestrates the entire simulation process, from initialization to result summarization.

Dependencies:

yaml, random, simulation.engine, simulation.scheduler, utils.data_generation, models.demand, models.insight, models.trip.

2. config.yaml
Purpose: Defines simulation parameters, including city details, driver and demand settings, optimization weights, and visualization options.
Structure:

city: Name of the city (e.g., "Hyderabad, India").
hotspots: Number of hotspot nodes (2000).
h3_resolution: H3 resolution for geospatial indexing (7).
num_drivers: Number of drivers (5).
steps_per_epoch: Simulation steps per epoch (150, equivalent to 2.5 hours).
epochs: Number of simulation epochs (2).
insight_interval: Frequency of insight generation (every 15 steps).
demand_rate_per_step: New demands generated Colonies per step (5000).
max_trip_distance: Maximum trip distance in meters (5000).
num_demands: Total demands to generate (5000).
optimization: Parameters for the optimization model (e.g., lambda, alpha, p_range, s_range, r_range, M_min, beta, gamma, max_grid_distance).
visualization: Settings for animation and saving (e.g., animate, save, filename, interval).

Optimizations:

Flexible configuration allows easy tuning of simulation parameters.
H3 resolution is constrained to 7–10 to balance precision and performance.

3. simulation/engine.py
Purpose: Implements the core simulation logic, managing driver movements, demand updates, and metrics logging.
Logic:

Initializes the simulation with configuration, drivers, models, and road graph.
For each epoch:
Resets drivers to home locations and clears trip caches.
Generates offline insights at the start of the epoch.
Runs steps within the epoch, invoking the scheduler.
Logs driver positions and metrics (fulfilled trips, unfulfilled demands, revenue, driver utilization).


Optionally generates visualizations using position logs.
Summarizes results across all epochs.

Optimizations:

Uses a defaultdict for metrics to simplify logging.
Employs a coordinate transformer (pyproj.Transformer) for accurate geospatial conversions.
Validates driver positions to prevent plotting invalid coordinates.
Caches visualization outputs to avoid redundant computations.

Key Classes and Methods:

SimulationEngine:
__init__: Initializes simulation components and validates config.
run_epoch: Executes one epoch, resetting drivers and running steps.
log_step: Logs driver positions and metrics for each step.
summarize_results: Computes and prints average metrics.



Dependencies:

logging, collections.defaultdict, utils.visualization, os, pyproj.

4. simulation/scheduler.py
Purpose: Manages the simulation’s step-by-step execution, coordinating demand generation, insight updates, and trip assignments.
Logic:

Each step:
Generates new demand units at the specified rate.
Generates insights every insight_interval steps.
Allows drivers to decide actions (accept insights or wait).
Assigns trips to idle drivers based on demand in their H3 zone.
Advances driver movements and demand timers.



Optimizations:

Validates input parameters to ensure positive intervals and non-empty driver lists.
Uses H3-based zoning to reduce the search space for trip assignments.
Processes drivers sequentially to avoid conflicts in trip assignments.

Key Classes and Methods:

Scheduler:
__init__: Initializes with insight interval, demand rate, and drivers.
step: Executes one simulation step, coordinating demand, insights, and trips.



Dependencies:

logging, models.driver, models.insight, models.demand, models.trip, networkx.

5. models/demand.py
Purpose: Models passenger demand, generating and managing demand units with origin and destination nodes.
Logic:

DemandUnit: Represents a single demand with attributes like origin/destination nodes, coordinates, H3 cells, and wait time.
DemandModel:
Precomputes reachable destinations for hotspot nodes using Dijkstra’s algorithm.
Generates demands by selecting random origin-destination pairs within the maximum trip distance.
Reuses a fraction of past demands to simulate recurring patterns.
Updates demand timers and deactivates demands exceeding the maximum wait time.



Optimizations:

Precomputes reachable destinations to avoid repeated pathfinding.
Uses H3 indexing for efficient spatial queries.
Reuses demands to reduce generation overhead.

Key Classes and Methods:

DemandUnit:
__init__: Initializes a demand unit.
step: Increments wait time and deactivates if expired.


DemandModel:
__init__: Sets up the model with hotspots and graph.
generate_demand: Creates new demand units, reusing some past demands.
get_active_demands: Returns active demand units.
step: Advances timers for all demands.



Dependencies:

typing.List, random, networkx, osmnx, h3, shapely.geometry.

6. models/driver.py
Purpose: Models driver behavior, including movement, trip execution, and insight acceptance.
Logic:

Driver:
Tracks position, status (idle, on_trip, repositioning), revenue, and trip counts.
Decides actions based on insight availability and propensity.
Moves along paths using shortest-path routing.
Updates position and H3 cell after movements or trips.
Handles trip start/completion and revenue calculation.



Optimizations:

Caches home location and node for efficient resets.
Uses osmnx for accurate nearest-node lookups.
Validates H3 cells and coordinates to prevent errors.
Employs propensity-based insight acceptance to simulate driver behavior.

Key Methods:

is_available: Checks if the driver is idle.
update_h3_cell: Updates the driver’s H3 cell based on lat/lon.
update_node: Updates the driver’s current node and coordinates.
receive_insight: Sets a repositioning path to an insight node.
set_path: Sets the driver’s movement path.
start_trip: Initiates a trip for a demand.
complete_trip: Finalizes a trip and updates position.
step_along_path: Moves the driver along the path.
wait: Increments idle time.
decide_action: Decides whether to accept an insight or wait.

Dependencies:

random, networkx, osmnx, h3, logging, shapely.geometry, models.demand.

7. models/insight.py
Purpose: Generates insights for drivers, recommending high-demand zones using online and offline optimization.
Logic:

InsightModel:
Online insights: Suggests nearby nodes randomly for idle drivers.
Offline insights: Uses an optimization model to assign drivers to H3 zones based on demand, distance, and revenue.
Maintains a mapping of driver IDs to suggested nodes.



Optimizations:

Uses KDTree for spatial indexing when direct node mapping fails.
Combines hotspot-based and spatial-index-based node selection for robustness.
Employs optimization (ILP or simulated annealing) for offline assignments.
Filters nearby nodes to avoid trivial movements.

Key Methods:

__init__: Initializes with demand data and a random seed.
generate_insights: Generates online insights for idle drivers.
generate_offline_insights: Runs optimization to assign drivers to zones.
get_insight_for_driver: Retrieves the suggested node for a driver.

Dependencies:

typing.List, random, models.driver, models.demand, networkx, osmnx, h3, utils.optimization_manager, logging, collections.defaultdict, scipy.spatial, numpy.

8. models/trip.py
Purpose: Manages trip assignments, calculating paths and fares for drivers.
Logic:

TripModel:
Assigns trips to drivers based on H3 zone proximity and demand wait time.
Computes shortest paths for pickup and drop-off.
Calculates fares based on distance and fare rate.


assign_driver_trips: Helper function to assign a trip to a single driver.

Optimizations:

Caches distances to avoid redundant path computations.
Prioritizes demands with longer wait times to reduce unfulfilled demands.
Validates trip distances against the maximum allowed distance.
Processes assignments sequentially to prevent conflicts.

Key Methods:

TripModel:
__init__: Initializes with maximum distance and fare rate.
reset_cache: Clears the distance cache.
assign_trips: Assigns trips to drivers.


assign_driver_trips: Computes path and fare for a driver-demand pair.

Dependencies:

typing.List, typing.Tuple, models.driver, models.demand, networkx, logging, h3.

9. utils/data_generation.py
Purpose: Generates initial data for the simulation, including the city graph, hotspots, and drivers.
Logic:

load_city_graph: Downloads or loads a cached road network graph for the specified city.
get_hotspot_nodes: Selects random nodes as hotspots and computes their coordinates and H3 cells.
initialize_drivers: Places drivers at random nodes, with optional hotspot overlap.

Optimizations:

Caches the city graph to disk to avoid repeated downloads.
Uses osmnx for efficient graph generation and projection.
Ensures random selection with a fixed seed for reproducibility.

Key Functions:

load_city_graph: Loads or downloads the city’s road network.
get_hotspot_nodes: Generates hotspot nodes with geospatial metadata.
initialize_drivers: Creates driver instances with initial positions.

Dependencies:

osmnx, random, models.driver, h3, shapely.geometry, pickle, os.

10. utils/optimization_manager.py
Purpose: Implements the optimization logic for assigning drivers to demand zones using Integer Linear Programming (ILP) or Simulated Annealing (SA).
Logic:

DataLoader: Prepares data for optimization, including driver H3 cells, zone demands, and optimization parameters.
DynamicMCalculator: Dynamically calculates the minimum number of zones to cover based on demand and driver count.
ILPSolver: Formulates and solves an ILP to maximize a weighted objective of offline (demand-based) and online (surge-based) goals.
SimulatedAnnealingSolver: Provides a heuristic alternative to ILP for faster computation.
OptimizationManager: Coordinates the optimization process, choosing between ILP and SA.

Optimizations:

Uses H3 grid distance for efficient spatial calculations.
Implements a feasible pair set to reduce the ILP problem size.
Dynamically updates affinity (p), surge (s), and revenue priority (r_priority) based on real-time data.
Employs simulated annealing for faster solutions in large-scale scenarios.
Validates H3 cells and distances to prevent errors.

Key Classes and Methods:

DataLoader:
__init__: Initializes with drivers, zone demands, and config.
_compute_haversine_distance: Calculates distance between coordinates and zone centroids.
_generate_feasible_pairs: Creates a set of valid driver-zone pairs.
update_dynamic_data: Updates optimization parameters.


DynamicMCalculator:
calculate_M: Computes the minimum number of zones to cover.


ILPSolver:
formulate: Sets up the ILP problem.
solve: Solves the ILP and returns assignments.


SimulatedAnnealingSolver:
objective_value: Computes the objective for a solution.
is_valid: Checks solution validity.
generate_initial_solution: Creates an initial feasible solution.
get_neighbor: Generates a neighboring solution.
solve: Runs the SA algorithm.


OptimizationManager:
run_optimization: Executes the optimization process.



Dependencies:

numpy, pulp, time, math, typing, logging, h3, models.driver.

11. utils/visualization.py
Purpose: Generates animations of driver movements across the city graph.
Logic:

Uses matplotlib and osmnx to plot the road network and driver paths.
Animates driver movements based on position logs, with each driver represented by a unique colored line.
Supports saving animations as MP4 or GIF files and optional live display.

Optimizations:

Uses line plots instead of scatter plots to show continuous paths.
Validates position coordinates to avoid plotting errors.
Supports both MP4 and GIF formats for flexibility.
Closes plots when not displayed to free resources.

Key Functions:

animate_from_log: Creates an animation from the driver position log.

Dependencies:

matplotlib.pyplot, matplotlib.animation, matplotlib.cm, osmnx, logging.

System-Wide Optimizations

Geospatial Efficiency: Uses H3 indexing to partition the city into hexagonal cells, reducing the complexity of spatial queries and trip assignments.
Graph-Based Routing: Leverages networkx and osmnx for efficient shortest-path calculations on the road network.
Caching: Caches city graphs, distance calculations, and visualization outputs to minimize redundant computations.
Dynamic Optimization: Balances offline (future demand) and online (current surge) objectives using a weighted ILP or SA.
Reusability: Reuses past demands to simulate recurring patterns, reducing generation overhead.
Error Handling: Includes robust validation and logging to handle invalid coordinates, H3 cells, and optimization infeasibility.

Assumptions and Limitations

Assumes a static road network loaded via osmnx.
H3 resolution is fixed per simulation, limiting flexibility in spatial granularity.
ILP optimization may be slow for large numbers of drivers or zones; SA provides a faster alternative.
Driver behavior is simplified (e.g., fixed insight propensity).
Visualization assumes valid UTM coordinates within a specific range.

Future Improvements

Parallelize trip assignments for faster processing.
Incorporate real-time traffic data into path calculations.
Add support for dynamic H3 resolution adjustments.
Enhance visualization with demand heatmaps or trip annotations.
Implement more sophisticated driver behavior models (e.g., learning-based propensity).

Usage

Ensure dependencies (osmnx, h3, networkx, pulp, numpy, matplotlib, pyproj) are installed.
Configure config.yaml with desired parameters.
Run main.py to execute the simulation.
Check logs and output files (e.g., driver_animation.gif) for results and visualizations.

