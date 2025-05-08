import pickle
import numpy as np
from tqdm import tqdm
from path_finder import get_trajectory_pkl_paths, save_trajectory_smoothed_pkl_path
from datetime import datetime
from scipy.signal import savgol_filter

def load_trajectory_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_trajectory_dict(trajectory_dict, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(trajectory_dict, f)

def compute_max_displacement(traj):
    """Compute the maximum pairwise Euclidean distance in the trajectory."""
    if len(traj) < 2:
        return 0.0
    coords = np.array([[point[1], point[2]] for point in traj])  # Assuming (timestamp, x, y)
    dists = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)
    return np.max(dists)

def compute_total_traveled_distance(traj):
    """Compute total distance traveled along the trajectory."""
    if len(traj) < 2:
        return 0.0
    coords = np.array([[point[1], point[2]] for point in traj])
    diffs = coords[1:] - coords[:-1]  # consecutive differences
    dists = np.linalg.norm(diffs, axis=1)  # Euclidean distance per step
    return np.sum(dists)

def compute_time_difference_in_seconds(t1, t2):
    """Compute the time difference between two timestamp strings in seconds."""
    dt1 = datetime.fromisoformat(t1)
    dt2 = datetime.fromisoformat(t2)
    delta = dt2 - dt1
    return delta.total_seconds()

def remove_stationaries(trajectory_dict, threshold):
    cleaned_dict_stationary = {}
    removed_agents = []

    for agent_id, traj in tqdm(trajectory_dict.items(), desc="Checking agents"):
        max_disp = compute_max_displacement(traj)
        if max_disp > threshold:
            cleaned_dict_stationary[agent_id] = traj
        else:
            removed_agents.append(agent_id)

    print(f"Removed {len(removed_agents)} stationary agents.")
    
    return cleaned_dict_stationary

def remove_spiky_trajectories(trajectory_dict, max_speed_threshold=10.0, max_jump_distance_threshold=6.0):
    """
    Remove trajectories that have unrealistic speeds or large jumps between consecutive points.

    Parameters:
    - max_speed_threshold: maximum allowed speed (m/s)
    - max_jump_distance: maximum allowed jump distance (m)
    """
    cleaned_dict = {}
    removed_velocity_agents = []
    removed_distance_agents = []

    for agent_id, traj in tqdm(trajectory_dict.items(), desc="Checking for spiky movements"):
        coords = np.array([[point[1], point[2]] for point in traj])
        timestamps = np.array([point[0] for point in traj])

        if len(coords) < 2:
            cleaned_dict[agent_id] = traj
            continue

        # Compute distances between consecutive points
        diffs = coords[1:] - coords[:-1]
        dists = np.linalg.norm(diffs, axis=1)

        # Compute time differences (in seconds)
        time_diffs = np.array([
            compute_time_difference_in_seconds(timestamps[i], timestamps[i+1])
            for i in range(len(timestamps) - 1)
        ])
        time_diffs = np.clip(time_diffs, 1e-6, None)  # avoid division by zero

        # Compute velocities between consecutive timesteps
        velocities = dists / time_diffs

        if np.any(dists > max_jump_distance_threshold):
            removed_distance_agents.append(agent_id)
        elif np.any(velocities > max_speed_threshold):
            #cleaned_dict[agent_id] = traj
            removed_velocity_agents.append(agent_id)
        else:
            cleaned_dict[agent_id] = traj

    print(f"Removed {len(removed_distance_agents)} agents with spiky distance movements.")
    print(f"Removed {len(removed_velocity_agents)} agents with spiky velocity movements.")
    return cleaned_dict

def remove_too_short_trajectories(trajectory_dict, min_total_distance=7.0):
    cleaned_dict = {}
    removed_agents = []

    for agent_id, traj in tqdm(trajectory_dict.items(), desc="Checking total distance"):
        total_dist = compute_total_traveled_distance(traj)
        if total_dist >= min_total_distance:
            cleaned_dict[agent_id] = traj
        else:
            removed_agents.append(agent_id)

    print(f"Removed {len(removed_agents)} short-distance agents.")
    return cleaned_dict

def apply_savgol_filter(trajectory_dict, window_length=11, polyorder=3):
    """
    Apply Savitzky-Golay filter to each trajectory in the dictionary.
    
    Parameters:
        trajectory_dict (dict): Dictionary with agent_id as key and list of (timestamp_str, x, y) tuples.
        window_length (int): The length of the filter window (must be odd and >= polyorder + 2).
        polyorder (int): The order of the polynomial used to fit the samples.

    Returns:
        dict: A new dictionary with the same keys and smoothed (timestamp, x, y) tuples.
    """
    smoothed_dict = {}

    for agent_id, traj in tqdm(trajectory_dict.items(), desc="Smoothing trajectories:"):
        if len(traj) < window_length:
            # Too short to filter, just copy as-is
            smoothed_dict[agent_id] = traj
            continue

        timestamps = [datetime.fromisoformat(t[0]) for t in traj]
        xs = np.array([t[1] for t in traj])
        ys = np.array([t[2] for t in traj])

        # Apply Savitzky-Golay filter
        xs_smooth = savgol_filter(xs, window_length, polyorder)
        ys_smooth = savgol_filter(ys, window_length, polyorder)

        # Reassemble smoothed trajectory
        smoothed_traj = [(ts.isoformat(), float(x), float(y)) for ts, x, y in zip(timestamps, xs_smooth, ys_smooth)]
        smoothed_dict[agent_id] = smoothed_traj

    return smoothed_dict

def main():
    input_path_list = get_trajectory_pkl_paths()

    # ------------ CONFIG ------------
    i = 0 # List element for file selection (0-5)
    min_total_distance = 10.0               # Minimum total distance of trajectory (m)
    max_speed_threshold = 10.0              # Maximum velocity for agent movement at any point (m/s)
    max_jump_distance_threshold = 5.0       # Maximum distance between two consecutive data points in a trajectory (m)
    window_length = 11                      # Sliding window length for Savitzky-Golay filter (datapoints)
    polyorder = 3                           # Polynomial order for savgol filter. Lower = smoother, Higher = sharper
    max_pairwise_distance_threshold = 0.5   # Maximum pairwise euclidean distance between all trajectory datapoints (m)
    # --------------------------------

    input_path = input_path_list[i]
    output_path = save_trajectory_smoothed_pkl_path(input_path)

    # LOAD
    print("Loading trajectory data...")
    traj_dict = load_trajectory_dict(input_path)

    # REMOVE SHORT TRAJECTORIES 1st TIME
    print("Removing too short trajectories BEFORE smoothing...")
    cleaned_dict = remove_too_short_trajectories(traj_dict, min_total_distance=min_total_distance)

    # SMOOTH MICRO JITTER IN TRAJECTORIES
    print("Applying Savitzky-Golay filter to smooth high-frequency trajectory jitter...")
    cleaned_dict = apply_savgol_filter(cleaned_dict, window_length=window_length, polyorder=polyorder)

    # REMOVE TRAJECTORIES WITH SPIKY MOVEMENT (BASED ON UNNATURAL VELOCITIES AND DISTANCE JUMPS)
    print("Removing spiky movements")
    cleaned_dict = remove_spiky_trajectories(cleaned_dict, max_speed_threshold=max_speed_threshold, max_jump_distance_threshold=max_jump_distance_threshold)

    # REMOVE SHORT TRAJECTORIES 2nd TIME
    print("Removing too short trajectories AFTER smoothing...")
    cleaned_dict = remove_too_short_trajectories(cleaned_dict, min_total_distance=min_total_distance)

    # REMOVE STATIONARY TRAJECTORIES
    print("Removing stationary agents...")
    cleaned_dict = remove_stationaries(cleaned_dict, threshold=max_pairwise_distance_threshold)

    print(f"---Removed a total of {len(traj_dict) - len(cleaned_dict)} trajectories out of {len(traj_dict)}.---")
    print(f"---The dictionary now contains {len(cleaned_dict)} unique trajectories.---")

    print(f"Saving cleaned data to '{output_path}'")
    save_trajectory_dict(cleaned_dict, output_path)

if __name__ == "__main__":
    main()