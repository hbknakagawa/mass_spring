import os
from utils.data_editor import process_csv_to_numpy

# python -m src.generate_dataset

INPUT_DIR = "data"
OUTPUT_DIR = "processed_datasets"
SEQ_LEN = 10        # LSTMなどの入力系列長（過去Nステップ）

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_id = 1  # 保存用の連番ID（ファイル名に使う）

    # data/ フォルダ内の各サブフォルダ（例: data/1, data/2, ...）を走査
    for folder in sorted(os.listdir(INPUT_DIR)):
        folder_path = os.path.join(INPUT_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        csv_path = os.path.join(folder_path, "data.csv")

        if os.path.exists(csv_path):
            print(f"▶ Processing {csv_path} ...")
            # 正規化 + 時系列切り出し + npy保存
            process_csv_to_numpy(
                csv_path=csv_path,
                output_dir=OUTPUT_DIR,
                file_id=file_id,
                seq_len=SEQ_LEN,
                input_cols=["Drive_Pos [x(t)]", "Mass_Pos [p(t)]"],
                target_cols=["Drive_Pos [x(t)]"]
            )

            file_id += 1

if __name__ == "__main__":
    main()
