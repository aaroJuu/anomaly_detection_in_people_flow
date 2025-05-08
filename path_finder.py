from pathlib import Path

"""Returns a list of full paths to all CSV/pkl files in the respective directory.
    Assumes that a data directories are in the same root directory as this repository. Like so:

├── root
    ├── anomaly_detection_in_people_flow/
    │   └── path_finder.py
    ├── raw_csvs/
    │   ├── data1.csv
    │   ├── data2.csv
    │   └── ...
    ├── clean_csvs/
    │   ├── clean_data1.csv
    │   ├── clean_data2.csv
        └── ...
"""

def get_raw_csv_file_paths():
    current_path = Path(__file__).resolve()
    raw_csvs_dir = current_path.parent.parent / "raw_csvs"
    return list(raw_csvs_dir.glob("*.csv"))

def save_clean_csv_file_path(input_path):
    current_path = Path(__file__).resolve()
    output_dir = current_path.parent.parent / "clean_csvs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = input_path.stem + "_clean.csv"
    output_path = output_dir / output_name
    return output_path

def get_clean_csv_file_paths():
    current_path = Path(__file__).resolve()
    clean_csvs_dir = current_path.parent.parent / "clean_csvs"
    return list(clean_csvs_dir.glob("*.csv"))

def save_trajectory_pkl_path(input_path):
    current_path = Path(__file__).resolve()
    output_dir = current_path.parent.parent / "trajectory_pickles"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = input_path.stem + "_trajectories.pkl"
    output_path = output_dir / output_name
    return output_path

def get_trajectory_pkl_paths():
    current_path = Path(__file__).resolve()
    traj_pkl_dir = current_path.parent.parent / "trajectory_pickles"
    return list(traj_pkl_dir.glob("*.pkl"))

def save_trajectory_smoothed_pkl_path(input_path):
    current_path = Path(__file__).resolve()
    output_dir = current_path.parent.parent / "trajectory_smoothed_pickles"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = input_path.stem + "_smoothed.pkl"
    output_path = output_dir / output_name
    return output_path

def get_trajectory_smoothed_pkl_paths():
    current_path = Path(__file__).resolve()
    traj_pkl_dir = current_path.parent.parent / "trajectory_smoothed_pickles"
    return list(traj_pkl_dir.glob("*.pkl"))