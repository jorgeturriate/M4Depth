import os
import pandas as pd
import argparse

def reduce_csvs(input_dir, output_dir, samples_per_csv=20):
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    print(f"Reducing {len(csv_files)} CSV files from {input_dir} -> {output_dir}")

    for csv_file in csv_files:
        file_path = os.path.join(input_dir, csv_file)
        df = pd.read_csv(file_path)

        # Drop duplicate indices if they exist
        df = df.reset_index(drop=True)

        total_samples = len(df)
        if total_samples <= samples_per_csv:
            print(f"[SKIP] {csv_file}: has only {total_samples} samples, less than or equal to target {samples_per_csv}. Copying full file.")
            df_reduced = df
        else:
            # Sample uniformly along the sequence, preserving temporal logic
            step = max(total_samples // samples_per_csv, 1)
            df_reduced = df.iloc[::step].head(samples_per_csv)

        output_path = os.path.join(output_dir, csv_file)
        df_reduced.to_csv(output_path, index=False)
        print(f"[OK] Saved {len(df_reduced)} samples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce KITTI CSV files for faster training.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with original CSV files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save reduced CSV files.")
    parser.add_argument("--samples_per_csv", type=int, default=20, help="Number of samples per sequence CSV.")
    args = parser.parse_args()

    reduce_csvs(args.input_dir, args.output_dir, args.samples_per_csv)
