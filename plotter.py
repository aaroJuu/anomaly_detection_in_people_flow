import pickle
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from path_finder import get_trajectory_smoothed_pkl_paths, get_trajectory_pkl_paths

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

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('1000 random trajectories')
    plt.grid(True)
    if sample_size is None or sample_size <= 20:
        plt.legend(fontsize='small')  # only if few trajectories, else legend gets messy
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
    smooth_path_list = get_trajectory_smoothed_pkl_paths() # Cleaned
    raw_path_list = get_trajectory_pkl_paths() # Not cleaned
    
    # ------ CONFIG INPUT PATH AND SAMPLE SIZE ------
    i = 0 # Configure list element for file selection (0-5)
    sample_size = 150 # Set to None to plot all, or an integer to plot a random sample
    random_seed = 4
    # -----------------------------------------------

    raw_path = raw_path_list[i] # Raw
    smooth_path = smooth_path_list[i]  # Smooth

    print("Loading raw trajectory data...")
    raw_dict = load_trajectory_dict(raw_path)
    print("Loading smoothed trajectory data...")
    smooth_dict = load_trajectory_dict(smooth_path)

    #compare_trajectories_singular(raw_dict, smooth_dict, sample_size=sample_size, random_seed=random_seed)
    compare_trajectories_multiple(raw_dict, smooth_dict, sample_size=sample_size, random_seed=random_seed)

    #plot_trajectories(smooth_dict, sample_size=sample_size, random_seed=random_seed)
    #plot_trajectories(raw_dict, sample_size=sample_size, random_seed=random_seed)

if __name__ == "__main__":
    main()
