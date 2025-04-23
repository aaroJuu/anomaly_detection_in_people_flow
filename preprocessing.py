import pandas as pd
from pathlib import Path
from collections import defaultdict

def get_raw_csv_file_paths():
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

def get_clean_csv_file_paths():
    current_path = Path(__file__).resolve()
    clean_csvs_dir = current_path.parent.parent / "clean_csvs"
    return list(clean_csvs_dir.glob("*.csv"))

def clean_csv_file(raw_csv_path, chunk_size=100000):
    """Removes unnecessary columns (floor_name, zone_name) and decimals.
       Writes new csv files to clean_csvs/."""
    
    current_path = Path(__file__).resolve()
    output_dir = current_path.parent.parent / "clean_csvs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = raw_csv_path.stem + "_clean.csv"
    output_path = output_dir / output_name

    use_columns = ["time", "agent_id", "x_axis", "z_axis"]
    float_columns = ["x_axis", "z_axis"]

    cleaned_chunks = []

    print("Cleaning file (Removing extra columns and decimals)")
    for chunk in pd.read_csv(raw_csv_path, usecols=use_columns, chunksize=chunk_size):
        chunk[float_columns] = chunk[float_columns].round(3) # Round float columns to 3 decimals
        cleaned_chunks.append(chunk)

    # Combine cleaned chunks
    print("Concatenating chunks...")
    full_df = pd.concat(cleaned_chunks, ignore_index=True)

    # Sort all data by agent_id and time
    print("Sorting full DataFrame by agent_id and time...")
    full_df.sort_values(by=["agent_id", "time"], inplace=True)

    # Save final output
    print("Writing to CSV...")
    full_df.to_csv(output_path, index=False)

    print(f"Cleaned file saved to: {output_path}")


if __name__ == "__main__":
    raw_csv_paths = get_raw_csv_file_paths()
    print(f"Found {len(raw_csv_paths)} CSV files:")
    for path in raw_csv_paths:
        print(" -", path)
    clean_csv_file(raw_csv_paths[0])