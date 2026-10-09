import pandas as pd
import argparse
import os
import sys

def select_specific_two(parquet_file_path, data_dir, output_dir, repeat_per_sample=64):
    try:
        df = pd.read_parquet(parquet_file_path, engine='pyarrow')
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    total_records = len(df)
    print(f"Total records in parquet: {total_records}")

    # We want to find index 999 and 382
    target_idx_1 = 999
    target_idx_2 = 382
    
    row1 = None
    row2 = None
    
    print(f"Searching for indices {target_idx_1} and {target_idx_2}...")

    # Iterate to find the specific rows
    # Assuming 'extra_info' column contains a dictionary/struct with 'index' key
    for i in range(total_records):
        try:
            # Get the value of extra_info for the current row
            extra_info = df.iloc[i]['extra_info']
            
            # Check index
            # extra_info might be a dict or a struct depending on how it was saved
            current_idx = None
            if isinstance(extra_info, dict):
                current_idx = extra_info.get('index')
            else:
                # Try attribute access or item access if it's a pyarrow struct scalar
                try:
                    current_idx = extra_info['index']
                except:
                    pass
            
            if current_idx == target_idx_1:
                row1 = df.iloc[[i]]
                print(f"Found index {target_idx_1} at row {i}")
            elif current_idx == target_idx_2:
                row2 = df.iloc[[i]]
                print(f"Found index {target_idx_2} at row {i}")
                
            if row1 is not None and row2 is not None:
                break
        except Exception as e:
            # print(f"Error processing row {i}: {e}")
            continue

    if row1 is None:
        print(f"Error: Could not find record with index {target_idx_1}")
        return
    if row2 is None:
        print(f"Error: Could not find record with index {target_idx_2}")
        return
    
    indices_str = [str(target_idx_1), str(target_idx_2)]
    file_suffix = f"pi{indices_str[0]}_{indices_str[1]}"
    print(f"Selected Internal Indices: {indices_str}")

    # Total records
    total_count = repeat_per_sample * 2

    # Construct output path
    # Naming convention: dsr_sub_specific_2samples_{indices}_r{total_count}.parquet
    output_filename = f"dsr_sub_specific_2samples_{file_suffix}_r{total_count}.parquet"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_parquet_path = os.path.join(output_dir, output_filename)

    # Repeat each row repeat_per_sample times
    if repeat_per_sample > 0:
        df1 = pd.concat([row1] * repeat_per_sample, ignore_index=True)
        df2 = pd.concat([row2] * repeat_per_sample, ignore_index=True)
        result_df = pd.concat([df1, df2], ignore_index=True)
    else:
        result_df = pd.concat([row1, row2], ignore_index=True)

    # Save
    result_df.to_parquet(output_parquet_path)
    print(f"Saved {len(result_df)} records to {output_parquet_path}")
    
    # Print sample info
    print("-" * 80)
    for i, row in enumerate([row1, row2]):
        print(f"Sample prompt {i+1} (Index {indices_str[i]}):")
        try:
            # Assuming prompt is a list of dicts or similar structure
            print(row.iloc[0]['prompt'][0]['content'])
        except:
            print("Could not print prompt.")
        print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select specific data points (999, 382) from parquet file.')
    parser.add_argument("--data_dir", type=str, default="data/train/one_shot_rlvr", help="Path to the data directory")
    parser.add_argument("--parquet_file_name", type=str, default="dsr_sub.parquet", help="Path to the parquet file")
    # Default 64 per sample -> 128 total
    parser.add_argument('--repeat_per_sample', type=int, default=64, help='Number of times to repeat each sample')
    
    args = parser.parse_args()

    data_dir = args.data_dir
    parquet_file_path = os.path.join(data_dir, args.parquet_file_name)
    
    if not os.path.exists(parquet_file_path):
        print(f"File not found: {parquet_file_path}")
        print("Please check the path or filename.")
    else:
        output_dir = "data/train/one_shot_rlvr"
        select_specific_two(parquet_file_path, data_dir, output_dir, args.repeat_per_sample)
