import os
import pandas as pd

# === CONFIGURATION ===
base_dir = "midair/train_data"
train_dirs = ["Kite_training", "PLE_training"]
seq_len = 4
db_seq_len = 8
stride = 10  # How many frames to skip between sequences
output_suffix = "_reduced"  # Appended to the filename

def reduce_csv_sequence(file_path, seq_len, db_seq_len, stride):
    df = pd.read_csv(file_path)
    total_seq_len = max(seq_len, db_seq_len)

    if len(df) < total_seq_len:
        return None

    reduced_rows = []
    for start_idx in range(0, len(df) - total_seq_len + 1, stride):
        sequence = df.iloc[start_idx:start_idx + total_seq_len]
        if len(sequence) == total_seq_len:
            reduced_rows.append(sequence)

    if not reduced_rows:
        return None

    reduced_df = pd.concat(reduced_rows, ignore_index=True)
    return reduced_df

def main():
    for train_dir in train_dirs:
        dir_path = os.path.join(base_dir, train_dir)
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".csv"):
                    csv_path = os.path.join(root, file)
                    print(f"Processing: {csv_path}")

                    reduced_df = reduce_csv_sequence(csv_path, seq_len, db_seq_len, stride)
                    if reduced_df is not None:
                        new_name = file.replace(".csv", f"{output_suffix}.csv")
                        output_path = os.path.join(root, new_name)
                        reduced_df.to_csv(output_path, index=False)
                        print(f"Saved: {output_path}")
                    else:
                        print(f"Skipped (too short): {csv_path}")

if __name__ == "__main__":
    main()
