import pandas as pd
import argparse
import os


def _extract_internal_index(extra_info):
    if isinstance(extra_info, dict):
        return extra_info.get('index')
    return None


def random_select_two(parquet_file_path, data_dir, repeat_per_sample=64, pi_a=1, pi_b=289):
    try:
        df = pd.read_parquet(parquet_file_path, engine='pyarrow')
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    total_records = len(df)
    if total_records < 2:
        print("Error: Parquet file has fewer than 2 records.")
        return

    # Select 2 specific samples by internal index (extra_info['index'])
    if 'extra_info' not in df.columns:
        print("Error: Column 'extra_info' not found in parquet.")
        return

    internal_index_series = df['extra_info'].apply(_extract_internal_index)

    idx_a = internal_index_series[internal_index_series == pi_a].index
    idx_b = internal_index_series[internal_index_series == pi_b].index

    if len(idx_a) == 0 or len(idx_b) == 0:
        missing = []
        if len(idx_a) == 0:
            missing.append(f"pi{pi_a}")
        if len(idx_b) == 0:
            missing.append(f"pi{pi_b}")
        print(f"Error: Could not find requested internal indices: {', '.join(missing)}")
        return

    # If duplicates exist, take the first occurrence
    selected_indices = [int(idx_a[0]), int(idx_b[0])]
    row1 = df.iloc[[selected_indices[0]]]
    row2 = df.iloc[[selected_indices[1]]]

    indices_str = [str(pi_a), str(pi_b)]
    file_suffix = f"pi{indices_str[0]}_{indices_str[1]}"

    print(f"Selected Row Indices: {selected_indices}")
    print(f"Selected Internal Indices: {indices_str}")

    # Total records
    total_count = repeat_per_sample * 2

    # Construct output path
    # Naming convention: dsr_sub_random_2samples_{indices}_r{total_count}.parquet
    output_filename = f"dsr_sub_random_2samples_{file_suffix}_r{total_count}.parquet"
    output_parquet_path = os.path.join(data_dir, output_filename)

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
            print(row.iloc[0]['prompt'][0]['content'])
        except:
            print("Could not print prompt.")
        print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select two specific data points (pi) from parquet file.')
    parser.add_argument("--data_dir", type=str, default="data/train/one_shot_rlvr", help="Path to the data directory")
    parser.add_argument("--parquet_file_name", type=str, default="dsr_sub.parquet", help="Path to the parquet file")
    # Default 64 per sample -> 128 total
    parser.add_argument('--repeat_per_sample', type=int, default=64, help='Number of times to repeat each sample')
    parser.add_argument('--pi_a', type=int, default=1, help='First internal index (extra_info.index), e.g. 1 for pi1')
    parser.add_argument('--pi_b', type=int, default=289, help='Second internal index (extra_info.index), e.g. 289 for pi289')
    
    args = parser.parse_args()

    data_dir = args.data_dir
    parquet_file_path = os.path.join(data_dir, args.parquet_file_name)
    
    random_select_two(
        parquet_file_path,
        data_dir,
        repeat_per_sample=args.repeat_per_sample,
        pi_a=args.pi_a,
        pi_b=args.pi_b,
    )
