import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from path_finder import get_clean_csv_file_paths, save_trajectory_pkl_path
import pickle

def build_trajectories(clean_csv_path):
    """
    Builds a dictionary of trajectories from a cleaned CSV file.
    Format: { agent_id: [(time, x, z), ...] }
    """
    use_columns = ["time", "agent_id", "x_axis", "z_axis"]
    df = pd.read_csv(clean_csv_path, usecols=use_columns)

    # Create trajectory dict
    trajectories = defaultdict(list)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Building trajectories from {clean_csv_path.name}"):
        agent_id = row["agent_id"]
        point = (row["time"], row["x_axis"], row["z_axis"])
        trajectories[agent_id].append(point)

    return trajectories

def save_trajectory_dict(trajectories, clean_csv_path):
    """
    Saves the trajectory dictionary as a pickle file under trajectory_pickles/
    """
    output_path = save_trajectory_pkl_path(clean_csv_path)

    with open(output_path, "wb") as f:
        pickle.dump(trajectories, f)

    print(f"Trajectory dictionary saved to: {output_path}")
    return output_path

def main():
    clean_csv_paths = get_clean_csv_file_paths()         # list of clean_csv paths
    # ---- FILE CONFIG ----
    i = 0 # Configure list element for file selection (0-5)
    # ---------------------

    input_path = clean_csv_paths[i]
    traj_dict = build_trajectories(input_path)
    save_trajectory_dict(traj_dict, input_path)

if __name__ == "__main__":
    main()