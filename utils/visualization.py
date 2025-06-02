# utils/visualization.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import osmnx as ox
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

def animate_from_log(G, position_log, interval=300, save_path=None, show=False):
    if not position_log:
        logging.info("Empty position log, cannot generate animation")
        return

    logging.info(f"Position log size: {len(position_log)} steps, sample: {position_log[:2]}")
    if not G or not G.nodes:
        logging.info("Invalid or empty graph, cannot plot map")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    try:
        ox.plot_graph(G, ax=ax, show=False, close=False, node_size=0, edge_color="#BBBBBB", edge_linewidth=0.4)
        logging.info("Road network graph plotted successfully")
    except Exception as e:
        logging.info(f"Failed to plot graph: {e}")
        plt.close(fig)
        return

    driver_ids = sorted({driver_id for step in position_log for driver_id, _ in step})
    if not driver_ids:
        logging.info("No driver IDs found in position log")
        plt.close(fig)
        return

    logging.info(f"Driver IDs: {driver_ids}")
    cmap = cm.get_cmap("tab20", len(driver_ids))
    driver_to_color = {driver_id: cmap(i) for i, driver_id in enumerate(driver_ids)}

    # Initialize lines for each driver instead of scatter points
    line_dict = {
        driver_id: ax.plot([], [], '-', color=driver_to_color[driver_id], label=f"Driver {driver_id} Path", linewidth=2)[0]
        for driver_id in driver_ids
    }

    # Store position history for each driver
    position_history = {driver_id: {'x': [], 'y': []} for driver_id in driver_ids}

    def update(frame):
        logging.info(f"Updating frame {frame}")
        step_data = {driver_id: pos for driver_id, pos in position_log[frame]}
        for driver_id, line in line_dict.items():
            if driver_id in step_data:
                x, y = step_data[driver_id]
                if not (206952 <= x <= 247270 and 1914151 <= y <= 1942803):
                    # Clear line if position is invalid
                    line.set_data([], [])
                    logging.info(f"Step {frame}, Driver {driver_id}: Invalid position ({x}, {y})")
                else:
                    # Append valid position to history
                    position_history[driver_id]['x'].append(x)
                    position_history[driver_id]['y'].append(y)
                    # Update line with full history
                    line.set_data(position_history[driver_id]['x'], position_history[driver_id]['y'])
                    logging.info(f"Step {frame}, Driver {driver_id}: Plotted at ({x}, {y})")
            else:
                # Clear line if no data
                line.set_data([], [])
                logging.info(f"Step {frame}, Driver {driver_id}: No position data")
        ax.set_title(f"Driver Paths - Step {frame}")
        return line_dict.values()

    # Set axes limits after plotting graph
    ax.set_xlim(206952, 247270)
    ax.set_ylim(1914151, 1942803)
    logging.info("Set axes limits: xlim=(206952, 247270), ylim=(1914151, 1942803)")

    ax.legend(fontsize='x-small', loc="upper right")
    ani = animation.FuncAnimation(fig, update, frames=len(position_log), interval=interval, blit=False, repeat=False)
    logging.info(f"Created animation with {len(position_log)} frames")

    if save_path:
        logging.info(f"💾 Saving animation to {save_path}")
        try:
            if save_path.endswith(".mp4"):
                ani.save(save_path, writer='ffmpeg', dpi=100)
            elif save_path.endswith(".gif"):
                ani.save(save_path, writer='pillow', dpi=100)
            else:
                raise ValueError("Unsupported format: use .mp4 or .gif")
        except Exception as e:
            logging.info(f"Failed to save animation: {e}")

    if show:
        plt.show()
    else:
        plt.close(fig)

