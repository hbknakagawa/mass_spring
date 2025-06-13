# src/batch_data_collection.py

from utils.batch_generator import generate_batch_data

# python -m src.batch_data_collection

def collect_batch_data():
    base_params = {
        "m": 1.0,
        "k": 10.0,
        "c": 1.0,
        "l": 1.0,
        "dt": 0.1,
        "T": 30.0,
        "drive_type": "wavy_decay",  # "exp" or "wavy_decay" or 
    }

    num_samples = 50  # 生成するデータ数

    # "wavy_decay"モードの場合
    A_range = (0.1, 1.0)
    w_range = (1.0, 10.0)
    b_range = (0.1, 1.5)

    generate_batch_data(num_samples=num_samples, base_params=base_params, 
                        A_range=A_range, w_range=w_range, b_range=b_range)

if __name__ == "__main__":
    collect_batch_data()