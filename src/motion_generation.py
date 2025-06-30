import os
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import json

from model.RNN import RNNModel
from utils.loss import physics_step

# === パラメータ読み込み（正規化解除用）===
with open("processed_datasets/scaler.json", "r") as f:
    scaler = json.load(f)

def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val


# === ハイパーパラメータ ===
seq_len = 10
input_size = 2   # [x, p]
hidden_size = 64
output_size = 1
device = torch.device("cpu")

# === モデル読み込み ===
model_path = "logs/learning_logs/20250620_1501/models/model_300.pth"
model = RNNModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === 保存先ディレクトリ作成 ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("logs", "motion_logs", timestamp)
os.makedirs(log_dir, exist_ok=True)

# === 初期シーケンス読み込みと次元選択（x, pのみ）===
initial_seq = np.load("processed_datasets/input_001.npy")[0:1]  # (1, 10, 6)
x_seq = torch.tensor(initial_seq, dtype=torch.float32)


# === 動作生成 ===
generated_x_norm = []
generated_p_norm = []

for t in range(50):
    x_pred = torch.sigmoid(model(x_seq)).detach()
    generated_x_norm.append(x_pred.item())
    print(generated_x_norm)

    p_t = x_seq[:, -1, 1]
    v_t = torch.zeros_like(p_t)
    p_next = physics_step(x_pred[:, 0], p_t, v_t).unsqueeze(1)
    generated_p_norm.append(p_next.item())

    next_input = torch.cat([x_pred, p_next], dim=1).unsqueeze(1)
    x_seq = torch.cat([x_seq[:, 1:], next_input], dim=1)

# === 正規化解除（元スケールに戻す）===
x_min, x_max = scaler["Drive_Pos [x(t)]"]["min"], scaler["Drive_Pos [x(t)]"]["max"]
p_min, p_max = scaler["Mass_Pos [p(t)]"]["min"], scaler["Mass_Pos [p(t)]"]["max"]

print(x_min,x_max)

# 各時系列を逆正規化
generated_x = [denormalize(x, x_min, x_max) for x in generated_x_norm]
generated_p = [denormalize(p, p_min, p_max) for p in generated_p_norm]

print("###############################")
print(generated_x)

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

