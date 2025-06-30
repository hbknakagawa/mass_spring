import os
import json
import numpy as np
import pandas as pd
from utils.data_editor import process_csv_to_numpy, normalize_dataframe

INPUT_DIR = "data"
OUTPUT_DIR = "processed_datasets"
SEQ_LEN = 10

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_id = 1
    scaler_dict = {}

    # 正規化統計量を記録するためのリスト
    mins = []
    maxs = []

    for folder in sorted(os.listdir(INPUT_DIR)):
        folder_path = os.path.join(INPUT_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        csv_path = os.path.join(folder_path, "data.csv")

        if os.path.exists(csv_path):
            print(f"▶ Processing {csv_path} ...")

            # CSV読み込みと正規化
            df = pd.read_csv(csv_path)
            df_norm, stats = normalize_dataframe(
                df,
                exclude_cols=["Timestep", "Time [s]"],
                return_stats=True
            )

            mins.append(stats["min"])
            maxs.append(stats["max"])
            
            # .npy保存（正規化済みデータフレームを渡す）
            process_csv_to_numpy(
                df_norm=df_norm,
                file_id=file_id,
                output_dir=OUTPUT_DIR,
                seq_len=SEQ_LEN,
                input_cols=["Drive_Pos [x(t)]", "Mass_Pos [p(t)]"],
                target_cols=["Drive_Pos [x(t)]"]
            )

            file_id += 1

    # スケーラ平均計算（全ファイル平均 or 最初の1個）
    final_min = mins[0]
    final_max = maxs[0]
    # ↑ 必要に応じて np.mean(..., axis=0) に変更可

    # 保存
    scaler = {
        col: {"min": final_min[col], "max": final_max[col]}
        for col in final_min
    }

    with open(os.path.join(OUTPUT_DIR, "scaler.json"), "w") as f:
        json.dump(scaler, f, indent=2)

    print("✅ 正規化スケーラを保存しました → processed_datasets/scaler.json")

if __name__ == "__main__":
    main()


# import os
# from utils.data_editor import process_csv_to_numpy

# # python -m src.generate_dataset

# INPUT_DIR = "data"
# OUTPUT_DIR = "processed_datasets"
# SEQ_LEN = 10        # LSTMなどの入力系列長（過去Nステップ）

# def main():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     file_id = 1  # 保存用の連番ID（ファイル名に使う）

#     # data/ フォルダ内の各サブフォルダ（例: data/1, data/2, ...）を走査
#     for folder in sorted(os.listdir(INPUT_DIR)):
#         folder_path = os.path.join(INPUT_DIR, folder)
#         if not os.path.isdir(folder_path):
#             continue

#         csv_path = os.path.join(folder_path, "data.csv")

#         if os.path.exists(csv_path):
#             print(f"▶ Processing {csv_path} ...")
#             # 正規化 + 時系列切り出し + npy保存
#             process_csv_to_numpy(
#                 csv_path=csv_path,
#                 output_dir=OUTPUT_DIR,
#                 file_id=file_id,
#                 seq_len=SEQ_LEN,
#                 input_cols=["Drive_Pos [x(t)]", "Mass_Pos [p(t)]"],
#                 target_cols=["Drive_Pos [x(t)]"]
#             )

#             file_id += 1

# if __name__ == "__main__":
#     main()
