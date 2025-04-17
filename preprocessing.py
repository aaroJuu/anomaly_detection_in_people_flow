import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict

def get_csv_file_paths():
    """Returns full paths to all CSV files in the ../root/raw_csvs directory.
       Assumes that "raw_csvs" directory is in the same root directory
       as this repository. Like so:
    
    ├── root
        ├── anomaly_detection_in_people_flow/
        │   └── preprocessing.py
        └── raw_csvs/
            ├── data1.csv
            ├── data2.csv
            └── ...
    """
    current_path = Path(__file__).resolve()
    raw_csvs_dir = current_path.parent.parent / "raw_csvs"
    return list(raw_csvs_dir.glob("*.csv"))



if __name__ == "__main__":
    raw_csv_paths = get_csv_file_paths()
    print(f"Found {len(raw_csv_paths)} CSV files:")
    for path in raw_csv_paths:
        print(" -", path)