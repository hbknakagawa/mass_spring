import numpy as np
import pandas as pd
import os

def normalize_dataframe(df, exclude_cols=None, return_stats=False):
    """
    指定した列を除いて Min-Max 正規化を実行。
    - exclude_cols: 正規化しない列（例: 'Time [s]', 'Timestep'）
    - return_stats: Trueなら min/max を辞書で返す
    """
    if exclude_cols is None:
        exclude_cols = []

    cols_to_normalize = [col for col in df.columns if col not in exclude_cols]
    df_norm = df.copy()
    min_dict = {}
    max_dict = {}

    for col in cols_to_normalize:
        col_min = df[col].min()
        col_max = df[col].max()
        df_norm[col] = (df[col] - col_min) / (col_max - col_min + 1e-8)
        min_dict[col] = col_min
        max_dict[col] = col_max

    if return_stats:
        return df_norm, {"min": min_dict, "max": max_dict}
    else:
        return df_norm

def split_sequence(df, seq_len, input_cols, target_cols):
    """
    過去 seq_len ステップを入力、次の1ステップを出力とする形式に切り出し
    """
    inputs = []
    targets = []

    for i in range(len(df) - seq_len):
        x_seq = df.iloc[i:i+seq_len][input_cols].values
        y_seq = df.iloc[i+seq_len][target_cols].values
        inputs.append(x_seq)
        targets.append(y_seq)

    return np.array(inputs), np.array(targets)

def process_csv_to_numpy(csv_path=None,
                         df_norm=None,
                         output_dir=None,
                         file_id=1,
                         seq_len=10,
                         exclude_cols=["Timestep", "Time [s]"],
                         input_cols=["Drive_Pos [x(t)]", "Mass_Pos [p(t)]"],
                         target_cols=["Drive_Pos [x(t)]"]):
    """
    CSVまたはDataFrameを使って、正規化済みデータからnp.saveを行う。
    """
    import pandas as pd
    import numpy as np
    import os

    # CSVから読む or df_normがすでに渡されているか
    if df_norm is None:
        if csv_path is None:
            raise ValueError("csv_path または df_norm のどちらかが必要です")
        df = pd.read_csv(csv_path)
        df_norm = normalize_dataframe(df, exclude_cols=exclude_cols)

    # 入力列自動選択（必要であれば）
    if input_cols is None:
        input_cols = [col for col in df_norm.columns if col not in exclude_cols and col not in target_cols]

    # 時系列分割
    X, y = split_sequence(df_norm, seq_len, input_cols, target_cols)

    # 保存
    input_path = os.path.join(output_dir, f"input_{file_id:03d}.npy")
    target_path = os.path.join(output_dir, f"target_{file_id:03d}.npy")
    np.save(input_path, X)
    np.save(target_path, y)

    print(f"Saved: {input_path}, {target_path}")


# import numpy as np
# import pandas as pd
# import os

# def normalize_dataframe(df, exclude_cols=None):
#     """
#     指定した列を除いて Min-Max 正規化を実行。
#     - exclude_cols: ['Time [s]', 'Timestep'] など正規化しない列
#     """
#     if exclude_cols is None:
#         exclude_cols = []
#     cols_to_normalize = [col for col in df.columns if col not in exclude_cols]
    
#     df_norm = df.copy()
#     for col in cols_to_normalize:
#         col_min = df[col].min()
#         col_max = df[col].max()
#         df_norm[col] = (df[col] - col_min) / (col_max - col_min + 1e-8)
#     return df_norm

# def split_sequence(df, seq_len, input_cols, target_cols):
#     """
#     過去 seq_len ステップを入力、次の1ステップを出力とする形式に切り出し
#     - df: 正規化済みDataFrame
#     - input_cols: 入力特徴量列名のリスト
#     - target_cols: 出力ターゲット列名のリスト
#     """
#     inputs = []
#     targets = []

#     for i in range(len(df) - seq_len):
#         x_seq = df.iloc[i:i+seq_len][input_cols].values
#         y_seq = df.iloc[i+seq_len][target_cols].values
#         inputs.append(x_seq)
#         targets.append(y_seq)

#     return np.array(inputs), np.array(targets)

# def process_csv_to_numpy(csv_path, output_dir, file_id,
#                          seq_len=10,
#                          exclude_cols=["Timestep", "Time [s]"],
#                          input_cols=["Drive_Pos [x(t)]", "Mass_Pos [p(t)]"],
#                          target_cols=["Drive_Pos [x(t)]"]):
#     """
#     1つのCSVファイルから、正規化 → 時系列切り出し → .npy保存まで行う
#     """
#     df = pd.read_csv(csv_path)

#     if input_cols is None:
#         # 自動的に入力列を推定（TimestepとTime以外全て）
#         input_cols = [col for col in df.columns if col not in exclude_cols and col not in target_cols]

#     df_norm = normalize_dataframe(df, exclude_cols=exclude_cols)
#     X, y = split_sequence(df_norm, seq_len, input_cols, target_cols)

#     # 保存
#     input_path = os.path.join(output_dir, f"input_{file_id:03d}.npy")
#     target_path = os.path.join(output_dir, f"target_{file_id:03d}.npy")
#     np.save(input_path, X)
#     np.save(target_path, y)

#     print(f"Saved: {input_path}, {target_path}")
