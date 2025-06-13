# utils/file_saver.py

import os
import pandas as pd
import matplotlib.pyplot as plt

def create_save_path(base_dir='data'):
    """
    保存先のディレクトリを自動生成（例: data/1）
    - 既存の最大数値ディレクトリに +1 した番号を使う
    """
    os.makedirs(base_dir, exist_ok=True)

    # data/ 以下の数字ディレクトリを調べて最大値を取得
    existing = [int(name) for name in os.listdir(base_dir) if name.isdigit()]
    new_index = max(existing) + 1 if existing else 1

    save_path = os.path.join(base_dir, str(new_index))
    os.makedirs(save_path, exist_ok=True)

    return save_path

def save_data_and_plot(save_path, t, x, vx, ax, p, v, a):
    """
    - データをCSVで保存
    - グラフをPNGで保存（位置・速度・加速度）
    """
    # --- CSV 保存 ---
    df = pd.DataFrame({
        'Timestep': list(range(len(t))),
        'Time [s]': t,
        'Drive_Pos [x(t)]': x,
        'Drive_Vel [vx(t)]': vx,
        'Drive_Acc [ax(t)]': ax,
        'Mass_Pos [p(t)]': p,
        'Mass_Vel [v(t)]': v,
        'Mass_Acc [a(t)]': a,
    })
    csv_path = os.path.join(save_path, 'data.csv')
    df.to_csv(csv_path, index=False)

    # --- グラフ 描写 + PNG 保存 ---
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, x, label='Drive x(t)', linestyle='--')
    plt.plot(t, p, label='Mass p(t)')
    plt.ylabel('Position [m]')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, vx, label='Drive vx(t)', linestyle='--')
    plt.plot(t, v, label='Mass v(t)')
    plt.ylabel('Velocity [m/s]')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, ax, label='Drive ax(t)', linestyle='--')
    plt.plot(t, a, label='Mass a(t)')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s²]')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    png_path = os.path.join(save_path, 'plot.png')
    plt.savefig(png_path)
    plt.close()

    print(f"Data saved at {csv_path}")
    print(f"Graph saved at {png_path}")

def save_config(save_path, config_dict):
    import json

    config_path = os.path.join(save_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"Params saved at {config_path}")
