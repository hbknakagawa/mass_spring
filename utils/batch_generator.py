# utils/batch_generator.py
import numpy as np
from utils.simulator import MassSpringSimulator
from utils.file_saver import create_save_path, save_data_and_plot, save_config

def generate_batch_data(num_samples, base_params, 
                        A_range=None, w_range=None, b_range=None):
    for _ in range(num_samples):
        drive_type = base_params["drive_type"]

        # 駆動パラメータのランダム生成
        if drive_type == "wavy_decay":
            A_val = np.random.uniform(*A_range)
            w_val = np.random.uniform(*w_range)
            b_val = np.random.uniform(*b_range)
            drive_param = {"A": A_val, "w": w_val, "b": b_val}
        elif drive_type == "exp":
            b_val = np.random.uniform(0.5, 3.0)
            drive_param = {"A": 1.0, "b": b_val}
        elif drive_type == "linear":
            v_val = np.random.uniform(0.01, 0.3)
            drive_param = {"v": v_val}
        else:
            drive_param = {"A": 1.0}  # fallback for other modes

        # シミュレーター初期化
        simulator = MassSpringSimulator(
            m=base_params["m"], k=base_params["k"], c=base_params["c"], l=base_params["l"],
            dt=base_params["dt"], T=base_params["T"]
        )

        # シミュレーション実行
        t, x, vx, ax, p, v, a = simulator.simulate(
            drive_type=base_params["drive_type"],
            drive_param=drive_param
        )

        # 保存
        save_path = create_save_path(base_dir="data")
        save_data_and_plot(save_path, t, x, vx, ax, p, v, a)
        save_config(save_path, {**base_params, "drive_param": drive_param})
