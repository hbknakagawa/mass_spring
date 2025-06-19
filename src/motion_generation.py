# motion_generation.py

import os
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

from model.RNN import RNNModel
from utils.loss import physics_step

# === ハイパーパラメータ ===
seq_len = 10
input_size = 2   # [x, vx, ax, p, v, a]
hidden_size = 64
output_size = 1
device = torch.device("cpu")

# === モデル読み込み ===
model_path = "logs/learning_logs/20250619_2114/models/model_100.pth"  # ★適宜変更
model = RNNModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === 保存先ディレクトリ作成 ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("logs", "motion_logs", timestamp)
os.makedirs(log_dir, exist_ok=True)

# === 初期シーケンス読み込み ===
initial_seq = np.load("processed_datasets/input_001.npy")[0:1]  # shape: (1, seq_len, 6)
x_seq = torch.tensor(initial_seq, dtype=torch.float32)

# === 動作生成 ===
generated_x = []
generated_p = []

for t in range(50):  # 時系列生成ステップ数
    x_pred = model(x_seq).detach()  # (1, 1)
    generated_x.append(x_pred.item())

    # p(t), v(t) を取り出す
    p_t = x_seq[:, -1, 1]
    v_t = torch.zeros_like(p_t)

    # 物理法則に基づいて p(t+1) を推定
    p_next = physics_step(x_pred[:, 0], p_t, v_t).unsqueeze(1)
    generated_p.append(p_next.item())

    next_input = torch.cat([x_pred, p_next], dim=1).unsqueeze(1) 
    x_seq = torch.cat([x_seq[:, 1:], next_input], dim=1)


# === プロット ===
plt.figure()
plt.plot(generated_x, label="x(t) [Drive Point]")
plt.plot(generated_p, label="p(t) [Mass Point]")
plt.xlabel("Time Step")
plt.ylabel("Position")
plt.title("Generated Motion")
plt.legend()
plt.grid()
plt.savefig(os.path.join(log_dir, "generated_motion.png"))
plt.close()

# === CSV保存 ===
df = pd.DataFrame({
    "step": list(range(1, len(generated_x)+1)),
    "x(t+1)": generated_x,
    "p(t+1)": generated_p,
})
df.to_csv(os.path.join(log_dir, "generated_motion.csv"), index=False)

print(f"✅ モーション生成完了: 保存先 {log_dir}")
