import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from preprocessing import get_clean_csv_file_paths
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
    current_path = Path(__file__).resolve()
    output_dir = current_path.parent.parent / "trajectory_pickles"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = clean_csv_path.stem.replace("_clean", "") + "_trajectories.pkl"
    output_path = output_dir / output_name

    with open(output_path, "wb") as f:
        pickle.dump(trajectories, f)

    print(f"Trajectory dictionary saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    clean_csv_paths = get_clean_csv_file_paths()         # list of clean_csv paths
    traj_dict = build_trajectories(clean_csv_paths[1])
    save_trajectory_dict(traj_dict, clean_csv_paths[1])

    # Optional: Inspect a sample
    sample_id = list(traj_dict.keys())[0]
    print(f"Sample trajectory for agent {sample_id}:\n{traj_dict[sample_id][:5]}")