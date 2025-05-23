"""import os
import pandas as pd

# === CONFIGURATION ===
base_dir = "midair/train_data"
train_dirs = ["Kite_training", "PLE_training"]
seq_len = 4
db_seq_len = 8
stride = 8  # How many frames to skip between sequences
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
    main()"""

"""import os
import pandas as pd

input_root = "midair"
output_root = "midair-reduced"
max_frames = 100

for root, _, files in os.walk(input_root):
    for file in files:
        if file.endswith(".csv"):
            input_path = os.path.join(root, file)

            # Ruta relativa y nueva ruta de salida
            rel_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Leer CSV con múltiples espacios como delimitador
            df = pd.read_csv(input_path, delim_whitespace=True)

            # Truncar y resetear id
            df_trunc = df.iloc[:max_frames].copy()
            df_trunc.iloc[:, 0] = range(len(df_trunc))

            # Guardar con separador de espacio
            df_trunc.to_csv(output_path, sep=' ', index=False)

            print(f"Saved: {output_path}  ({len(df_trunc)} rows)")"""

import os

input_root = "midair"
output_root = "midair_reduced"
max_lines = 101  # 1 header + 100 frames

for root, _, files in os.walk(input_root):
    for file in files:
        if file.endswith(".csv"):
            input_path = os.path.join(root, file)

            # Ruta relativa para reconstruir estructura
            rel_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Leer y escribir las primeras 101 líneas
            with open(input_path, "r") as infile:
                lines = infile.readlines()

            with open(output_path, "w") as outfile:
                outfile.writelines(lines[:max_lines])

            print(f"Saved: {output_path} ({min(len(lines)-1, 100)} frames)")


