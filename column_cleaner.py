import pandas as pd
from path_finder import get_raw_csv_file_paths, save_clean_csv_file_path

def clean_csv_file(raw_csv_path, chunk_size=100000):
    """Removes unnecessary columns (floor_name, zone_name) and decimals.
       Writes new csv files to clean_csvs/."""

    output_path = save_clean_csv_file_path(raw_csv_path)

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

def main():
    # ---- FILE CONFIG ----
    i = 0 # Configure list element for file selection (0-5)
    # ---------------------

    raw_csv_paths = get_raw_csv_file_paths()
    print(f"Found {len(raw_csv_paths)} CSV files:")
    for path in raw_csv_paths:
        print(" -", path)
    
    clean_csv_file(raw_csv_paths[i])

if __name__ == "__main__":
    main()