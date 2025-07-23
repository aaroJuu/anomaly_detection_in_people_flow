import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from path_finder import get_trajectory_smoothed_pkl_paths, get_trajectory_pkl_paths, get_removed_pkl_path, get_flowmap_pkl_paths
from obstacle_zones import obstacle_zones as oz
from flowmap import FlowMap
from cell import Cell

def load_trajectory_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def plot_trajectories(trajectory_dict, sample_size=100, random_seed=2):
    agent_ids = list(trajectory_dict.keys())
    
    if sample_size is not None:
        random.seed(random_seed)
        agent_ids = random.sample(agent_ids, min(sample_size, len(agent_ids)))
        print(f"Plotting {len(agent_ids)} random trajectories out of {len(trajectory_dict)} total.")
    else:
        print(f"Plotting all {len(agent_ids)} trajectories.")

    plt.figure(figsize=(12, 8))

    for agent_id in tqdm(agent_ids, desc="Plotting trajectories"):
        traj = trajectory_dict[agent_id]
        x = [point[1] for point in traj]  # Assuming (timestamp, x, y)
        y = [point[2] for point in traj]
        plt.plot(x, y, label=f'Agent {agent_id}', linewidth=1)

    # Draw dashed rectangles for each obstacle zone
    obstacle_zones = oz()
    for zone in obstacle_zones:
        x_min, x_max, y_min, y_max = zone
        plt.plot([x_min, x_max, x_max, x_min, x_min],
                 [y_min, y_min, y_max, y_max, y_min],
                 linestyle='--', color='red', linewidth=2)

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f"Plotting {len(agent_ids)} Random Trajectories", fontsize=16)
    plt.grid(True)
    if sample_size is None or sample_size <= 20:
        plt.legend(fontsize='small')  # only if few trajectories, else legend gets messy
    plt.show()

def plot_flowmap(flowmap, time_group='weekday', key=0):
    """
    Plot dominant flow direction for each cell at a specified time group,
    using color to indicate direction.

    Parameters:
        flowmap (FlowMap): Trained flow map object.
        time_group (str): One of ['hourly', 'time_block', 'weekday', 'daily']
        key: Time key (e.g., 14 for 14:00, '08-16', 2 for Tuesday, etc.)
    """

    # Define direction → color mapping
    direction_colors = {
        (1, 0): 'darkblue',     # E
        (1, 1): 'saddlebrown',  # NE
        (0, 1): 'red',          # N
        (-1, 1): 'orange',      # NW
        (-1, 0): 'limegreen',   # W
        (-1, -1): 'cyan',       # SW
        (0, -1): 'magenta',     # S
        (1, -1): 'purple',      # SE
    }

    X, Y, U, V, C = [], [], [], [], []

    for cell in flowmap.get_all_cells():
        dominant = cell.get_dominant_direction(time_group, key)
        if dominant:
            dx, dy = dominant
            if dx == 0 and dy == 0:
                continue  # skip stationary transitions

            # Center of the cell
            x = (cell.i + 0.5) * flowmap.cell_size
            y = (cell.j + 0.5) * flowmap.cell_size

            # Normalize vector to unit length
            norm = (dx ** 2 + dy ** 2) ** 0.5
            ux = dx / norm
            uy = dy / norm

            X.append(x)
            Y.append(y)
            U.append(ux)
            V.append(uy)
            C.append(direction_colors.get((dx, dy), 'black'))  # fallback to black if not recognized

    fig, ax = plt.subplots(figsize=(12, 8))

    # Quiver plot with normalized vectors
    ax.quiver(X, Y, U, V, color=C, angles='xy', scale_units='xy', scale=1.8, width=0.03,
            headlength=6, headwidth=4, headaxislength=6, minlength=0.01)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f"Flow Map Visualization — {time_group} = {key}", fontsize=16)
    ax.grid(True)

    # Plot obstacle zones
    for zone in oz():
        x_min, x_max, y_min, y_max = zone
        ax.plot([x_min, x_max, x_max, x_min, x_min],
                [y_min, y_min, y_max, y_max, y_min],
                linestyle='--', color='red', linewidth=2)

    # Set fixed view bounds and disable autoscaling
    ax.set_xlim(110, 190)
    ax.set_ylim(-180, -140)
    ax.set_aspect('equal')
    ax.set_autoscale_on(False)

    plt.show()

def plot_cell_distributions(cell, time_group='weekday', key=0, velocity_bins=24, v_range=(0, 3.0)):
    # --- Direction Distribution ---
    dir_dist = cell.get_direction_distribution(time_group, key)
    dir_labels = [str(d) for d in dir_dist.keys()]
    dir_values = list(dir_dist.values())

    # --- Velocity Histogram ---
    vel_hist = cell.get_velocity_histogram(time_group, key, bins=velocity_bins, range=v_range)
    bin_edges = np.linspace(v_range[0], v_range[1], velocity_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # --- Plotting ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Cell ({cell.i}, {cell.j}) — [{time_group} = {key}]", fontsize=14)

    # Direction subplot
    axs[0].bar(dir_labels, dir_values, color='skyblue')
    axs[0].set_title("Direction Distribution")
    axs[0].set_xlabel("Direction (dx, dy)")
    axs[0].set_ylabel("Probability")
    max_val = max(dir_values)
    axs[0].set_ylim(0, max_val*1.1)
    axs[0].grid(True)

    # Velocity subplot
    axs[1].bar(bin_centers, vel_hist, width=(bin_edges[1] - bin_edges[0]), color='salmon', edgecolor='black')
    axs[1].set_title("Velocity Distribution")
    axs[1].set_xlabel("Velocity (m/s)")
    axs[1].set_ylabel("Probability Density")
    max_val = max(vel_hist)
    axs[1].set_ylim(0, max_val*1.1)
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    plt.show()

def plot_cell_distribution_comparison(cell_tr, cell_obs, time_group='time_block', key='08-12', velocity_bins=24, v_range=(0, 3.0), results=None):
    # --- Find KL values if results provided ---
    dir_kl = vel_kl = vol_rd = None
    if results is not None:
        for r in results:
            if r['cell'] == (cell_tr.i, cell_tr.j) and r['time_group'] == time_group and r['key'] == key:
                dir_kl = r['direction_kl']
                vel_kl = r['velocity_kl']
                vol_rd = r['volume_rd']
                dir_kl_w = r['direction_kl_w']
                vel_kl_w = r['velocity_kl_w']
                vol_rd_w = r['volume_rd_w']
                vol_expected_avg = r['volume_expected_tr']
                vol_observed_avg = r['volume_expected_obs']
                break
    
    # --- Direction Distributions ---
    ordered_dirs = Cell.ALL_DIRECTIONS
    dir_dist_tr = cell_tr.get_direction_distribution(time_group, key)
    dir_dist_obs = cell_obs.get_direction_distribution(time_group, key)

    dir_values_tr = [dir_dist_tr.get(d, 0.0) for d in ordered_dirs]
    dir_values_obs = [dir_dist_obs.get(d, 0.0) for d in ordered_dirs]
    dir_labels = [str(d) for d in ordered_dirs]

    # --- Velocity Histograms ---
    vel_hist_tr = cell_tr.get_velocity_histogram(time_group, key, bins=velocity_bins, range=v_range)
    vel_hist_obs = cell_obs.get_velocity_histogram(time_group, key, bins=velocity_bins, range=v_range)

    bin_edges = np.linspace(v_range[0], v_range[1], velocity_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # --- Volume ---
    vol_tr = cell_tr.get_volume(time_group, key)
    vol_obs = cell_obs.get_volume(time_group, key)

    # --- Plotting ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Cell ({cell_tr.i}, {cell_tr.j}) — [{time_group} = {key}]", fontsize=14)

    # Direction subplot
    axs[0].bar(dir_labels, dir_values_tr, width=0.35, label='Trained (Q)', alpha=0.9, color='skyblue', edgecolor='black')
    axs[0].bar(dir_labels, dir_values_obs, width=0.35, label='Observed (P)', alpha=0.5, color='salmon', edgecolor='black')
    direction_title = "Direction Distribution"
    if dir_kl is not None:
        direction_title += f" (KLD = {dir_kl:.3f}, KLD_W = {dir_kl_w:.3f})"
    axs[0].set_title(direction_title)
    axs[0].set_xlabel("Direction (dx, dy)")
    axs[0].set_ylabel("Probability")
    axs[0].set_ylim(0, max(max(dir_values_tr), max(dir_values_obs)) * 1.1)
    axs[0].legend()
    axs[0].grid(True)

    # Velocity subplot
    axs[1].bar(bin_centers, vel_hist_tr, width=(bin_edges[1] - bin_edges[0]), label='Trained (Q)', alpha=0.9, color='skyblue', edgecolor='black')
    axs[1].bar(bin_centers, vel_hist_obs, width=(bin_edges[1] - bin_edges[0]), label='Observed (P)', alpha=0.5, color='salmon', edgecolor='black')
    velocity_title = "Velocity Distribution"
    if vel_kl is not None:
        velocity_title += f" (KLD = {vel_kl:.3f}, KLD_W = {vel_kl_w:.3f})"
    axs[1].set_title(velocity_title)
    axs[1].set_xlabel("Velocity (m/s)")
    axs[1].set_ylabel("Probability Density")
    axs[1].set_ylim(0, max(max(vel_hist_tr), max(vel_hist_obs)) * 1.1)
    axs[1].legend()
    axs[1].grid(True)

    # Add volume text below the plots
    volume_text = f"Volume — Trained (Q) (Total/Expected average): {vol_tr}/{vol_expected_avg:.1f}, Observed (P) (Total/Observed average): {vol_obs}/{vol_observed_avg:.1f}"
    if vol_rd is not None:
        volume_text += f", Relative Difference Log (+more/-less) = {vol_rd:.2f}"

    # Shrink layout and add text below
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.text(0.5, 0.01, volume_text, ha='center', fontsize=10)
    plt.show()

def plot_kl_distributions_by_timegroup(results):
    """
    Plot KL divergence histograms (direction and velocity) for each time group in a single figure.
    
    Parameters:
        results (list of dict): Output from kl_divergence_between_flowmaps()
    """
    # Organize data
    kl_data = defaultdict(lambda: {'direction': [], 'velocity': []})
    for r in results:
        group = r['time_group']
        if np.isfinite(r['direction_kl']):
            kl_data[group]['direction'].append(r['direction_kl'])
        if np.isfinite(r['velocity_kl']):
            kl_data[group]['velocity'].append(r['velocity_kl'])

    groups = list(kl_data.keys())
    n = len(groups)

    # Prepare figure with 2 rows: direction and velocity
    fig, axs = plt.subplots(2, n, figsize=(4 * n, 6), sharey=False)
    fig.suptitle("KL Divergence Distributions by Time Group", fontsize=16)

    # Share y-axis among first row
    for i in range(1, n):
        axs[0, i].sharey(axs[0, 0])
    # Share y-axis among second row
    for i in range(1, n):
        axs[1, i].sharey(axs[1, 0])

    for i, group in enumerate(groups):
        dir_vals = kl_data[group]['direction']
        vel_vals = kl_data[group]['velocity']

        # Direction subplot
        axs[0, i].hist(dir_vals, bins=60, color='skyblue', edgecolor='black', density=False)
        axs[0, i].set_title(f"{group} (Direction)", fontsize=10)
        axs[0, i].set_xlabel("KL Value")
        axs[0, i].grid(True)

        # Velocity subplot
        axs[1, i].hist(vel_vals, bins=60, color='salmon', edgecolor='black', density=False)
        axs[1, i].set_title(f"{group} (Velocity)", fontsize=10)
        axs[1, i].set_xlabel("KL Value")
        axs[1, i].grid(True)

    axs[0, 0].set_ylabel("Probability density")
    axs[1, 0].set_ylabel("Probability density")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def compare_trajectories_singular(raw_dict, smoothed_dict, sample_size=10, random_seed=2):
    """
    Compare raw and smoothed trajectories side-by-side for the same agents.

    Parameters:
        raw_dict (dict): Raw trajectory dictionary.
        smoothed_dict (dict): Smoothed trajectory dictionary.
        sample_size (int): Number of random agents to compare.
        random_seed (int): Random seed for reproducibility.
    """
    common_ids = list(set(raw_dict.keys()) & set(smoothed_dict.keys()))
    if not common_ids:
        print("No common agent IDs found between raw and smoothed dictionaries.")
        return

    random.seed(random_seed)
    selected_ids = random.sample(common_ids, min(sample_size, len(common_ids)))
    print(f"Comparing {len(selected_ids)} trajectories.")

    for agent_id in selected_ids:
        raw_traj = raw_dict[agent_id]
        smoothed_traj = smoothed_dict[agent_id]

        fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        fig.suptitle(f"Agent {agent_id}: Raw vs. Smoothed Trajectory", fontsize=14)

        raw_x = [pt[1] for pt in raw_traj]
        raw_y = [pt[2] for pt in raw_traj]
        axs[0].plot(raw_x, raw_y, label='Raw', color='blue', marker='o')
        axs[0].set_title("Raw Trajectory")
        axs[0].grid(True)

        smooth_x = [pt[1] for pt in smoothed_traj]
        smooth_y = [pt[2] for pt in smoothed_traj]
        axs[1].plot(smooth_x, smooth_y, label='Smoothed', color='green', marker='o')
        axs[1].set_title("Smoothed Trajectory")
        axs[1].grid(True)

        for ax in axs:
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')

        plt.tight_layout()
        plt.show()

def compare_trajectories_multiple(raw_dict, smoothed_dict, sample_size=100, random_seed=2):
    """
    Plot raw and smoothed trajectories side by side for a subset of agents.

    Parameters:
        raw_dict (dict): Dictionary of raw trajectories.
        smoothed_dict (dict): Dictionary of smoothed trajectories.
        sample_size (int): Number of agents to sample for comparison.
        random_seed (int): Random seed for reproducibility.
    """
    common_ids = list(set(raw_dict.keys()) & set(smoothed_dict.keys()))
    if not common_ids:
        print("No common agent IDs found between raw and smoothed dictionaries.")
        return

    random.seed(random_seed)
    selected_ids = random.sample(common_ids, min(sample_size, len(common_ids)))
    print(f"Comparing {len(selected_ids)} trajectories side by side.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

    # Plot raw
    axes[0].set_title("Raw Trajectories")
    for agent_id in tqdm(selected_ids, desc="Plotting raw trajectories"):
        traj = raw_dict[agent_id]
        x = [point[1] for point in traj]
        y = [point[2] for point in traj]
        axes[0].plot(x, y, linewidth=1)

    # Plot smoothed
    axes[1].set_title("Smoothed Trajectories")
    for agent_id in tqdm(selected_ids, desc="Plotting smoothed trajectories"):
        traj = smoothed_dict[agent_id]
        x = [point[1] for point in traj]
        y = [point[2] for point in traj]
        axes[1].plot(x, y, linewidth=1)

    for ax in axes:
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True)

    plt.suptitle(f"{len(selected_ids)} Random Trajectories (Raw vs Smoothed)", fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    #### TEMPORAL
    removed_path_dist = get_removed_pkl_path()[0]
    removed_path_obs = get_removed_pkl_path()[1]
    removed_path_vel = get_removed_pkl_path()[2]
    removed_dist_dict = load_trajectory_dict(removed_path_dist)
    removed_obs_dict = load_trajectory_dict(removed_path_obs)
    removed_vel_dict = load_trajectory_dict(removed_path_vel)
    #############
    
    # ------ CONFIG INPUT PATH AND SAMPLE SIZE ------
    i = 2 # Configure list element for file selection (0-5)
    sample_size = 10000 # Set to None to plot all, or an integer to plot a random sample
    random_seed = 76
    mode = 3        # Plotting mode: 0 = Trajectories,
                    #                1 = Flow map
                    #                2 = Cell-wise flow distribution
                    #                3 = Cell-wise flow distribution comparison
    t = 0 # File number for trained flowmap (ONLY IN MODE 3)
    o = 1 # File number for observed flowmap (ONLY IN MODE 3)
    # -----------------------------------------------

    if mode == 0: # TRAJECTORY PLOTTING
        raw_path_list = get_trajectory_pkl_paths() # Not cleaned    
        smooth_path_list = get_trajectory_smoothed_pkl_paths() # Cleaned

        raw_path = raw_path_list[i] # Raw
        smooth_path = smooth_path_list[i]  # Smooth

        print("Loading raw trajectory data...")
        raw_dict = load_trajectory_dict(raw_path)
        print("Loading smoothed trajectory data...")
        smooth_dict = load_trajectory_dict(smooth_path)

        # CHOOSE TRAJECTORY PLOTTING MODE
        #compare_trajectories_singular(raw_dict, smooth_dict, sample_size=sample_size, random_seed=random_seed)
        #compare_trajectories_multiple(raw_dict, smooth_dict, sample_size=sample_size, random_seed=random_seed)

        #plot_trajectories(removed_dist_dict, sample_size=sample_size, random_seed=random_seed)
        #plot_trajectories(removed_obs_dict, sample_size=sample_size, random_seed=random_seed)
        #plot_trajectories(removed_vel_dict, sample_size=sample_size, random_seed=random_seed)

        plot_trajectories(smooth_dict, sample_size=sample_size, random_seed=random_seed)
        #plot_trajectories(raw_dict, sample_size=sample_size, random_seed=random_seed)

    if mode == 1: # FLOW MAP PLOTTING
        flowmap_path_list = get_flowmap_pkl_paths()
        flowmap_path = flowmap_path_list[i]
        
        print("Loading flow map data...")
        flowmap = load_trajectory_dict(flowmap_path)

        # Visualization selection:
        # time_group: 'hourly',     key(int): 0-23
        #             'time_block', key(str): 00-08, 08-12, '12-16, '16-20', '20-24'
        #             'weekday',    key(int): 0-6 (Mon-Sun)
        #             'daily',      key: datetime.date()

        plot_flowmap(flowmap, time_group='time_block', key='00-08')
        plot_flowmap(flowmap, time_group='time_block', key='08-12')
        plot_flowmap(flowmap, time_group='time_block', key='12-16')
        plot_flowmap(flowmap, time_group='time_block', key='16-20')
        plot_flowmap(flowmap, time_group='time_block', key='20-24')
        """
        plot_flowmap(flowmap, time_group='hourly', key=0)
        plot_flowmap(flowmap, time_group='hourly', key=1)
        plot_flowmap(flowmap, time_group='hourly', key=2)
        plot_flowmap(flowmap, time_group='hourly', key=3)
        plot_flowmap(flowmap, time_group='hourly', key=4)
        plot_flowmap(flowmap, time_group='hourly', key=5)
        plot_flowmap(flowmap, time_group='hourly', key=6)
        plot_flowmap(flowmap, time_group='hourly', key=7)
        plot_flowmap(flowmap, time_group='hourly', key=8)
        plot_flowmap(flowmap, time_group='hourly', key=9)
        plot_flowmap(flowmap, time_group='hourly', key=10)
        plot_flowmap(flowmap, time_group='hourly', key=11)
        plot_flowmap(flowmap, time_group='hourly', key=12)
        plot_flowmap(flowmap, time_group='hourly', key=13)
        plot_flowmap(flowmap, time_group='hourly', key=14)
        plot_flowmap(flowmap, time_group='hourly', key=15)
        plot_flowmap(flowmap, time_group='hourly', key=16)
        plot_flowmap(flowmap, time_group='hourly', key=17)
        plot_flowmap(flowmap, time_group='hourly', key=18)
        plot_flowmap(flowmap, time_group='hourly', key=19)
        plot_flowmap(flowmap, time_group='hourly', key=20)
        plot_flowmap(flowmap, time_group='hourly', key=21)
        plot_flowmap(flowmap, time_group='hourly', key=22)
        plot_flowmap(flowmap, time_group='hourly', key=23)

        plot_flowmap(flowmap, time_group='weekday', key=0)
        plot_flowmap(flowmap, time_group='weekday', key=1)
        plot_flowmap(flowmap, time_group='weekday', key=2)
        plot_flowmap(flowmap, time_group='weekday', key=3)
        plot_flowmap(flowmap, time_group='weekday', key=4)
        plot_flowmap(flowmap, time_group='weekday', key=5)
        plot_flowmap(flowmap, time_group='weekday', key=6)
        """
    
    if mode == 2: # CELL-WISE FLOW DISTRIBUTION
        flowmap_path_list = get_flowmap_pkl_paths()
        flowmap_path = flowmap_path_list[i]
        
        print("Loading flow map data...")
        flowmap = load_trajectory_dict(flowmap_path)

        cells = flowmap.get_all_cells()

        for cell in cells:
            plot_cell_distributions(cell, time_group='time_block', key='08-12')

    if mode == 3: # CELL-WISE FLOW DISTRIBUTION COMPARISON
        flowmap_path_list = get_flowmap_pkl_paths()
        trained_flowmap_path = flowmap_path_list[t]
        observed_flowmap_path = flowmap_path_list[o]

        print("Loading flow map data...")
        trained_flowmap = load_trajectory_dict(trained_flowmap_path)
        observed_flowmap = load_trajectory_dict(observed_flowmap_path)

        for (i, j), trained_cell in trained_flowmap.grid.items():
            if (i, j) in observed_flowmap.grid:
                observed_cell = observed_flowmap.grid[(i, j)]
                plot_cell_distribution_comparison(trained_cell, observed_cell, time_group='hourly', key=14)

if __name__ == "__main__":
    main()
