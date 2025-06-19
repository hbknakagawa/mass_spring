import torch
import torch.nn as nn

def physics_step(x_next, p_t, v_t, m=1.0, k=10.0, c=1.0, l=1.0, dt=0.1, g=9.8):
    """
    モデルが出力した x(t+1) に基づいて、p(t+1) を物理シミュレーションで計算
    - x_next: モデル出力 (batch,)
    - p_t, v_t: 現在の位置と速度 (batch,)
    - 戻り値: p(t+1) のシミュレート結果
    """
    stretch = (p_t - x_next) + l
    a_t = (-k * stretch - c * v_t) / m - g
    v_next = v_t + a_t * dt
    p_next = p_t + v_next * dt
    return p_next

def custom_loss(x_pred, y_true, input_seq):
    """
    x_pred: (batch, output_size=1) — モデルが出力した x(t+1)
    y_true: (batch, output_size=1) — 実測の p(t+1)
    input_seq: (batch, seq_len, input_dim) — 入力シーケンス（過去のデータ）

    返り値:
        total_loss: MSE(p_next vs. y_true) + smoothness_penalty
    """
    # 最新ステップの p(t), v(t) を取得
    p_t = input_seq[:, -1, 1] 
    v_t = torch.zeros_like(p_t)

    # モデル予測の x(t+1) を使って p(t+1) を物理的に計算
    p_next = physics_step(x_pred[:, 0], p_t, v_t)  # shape: (batch,)

    # 正解の p(t+1)
    p_true = y_true[:, 0]

    # MSE（位置予測誤差）
    mse = nn.MSELoss()(p_next, p_true)

    # jerk最小（pの滑らかさ） — 2階差分
    p_seq = input_seq[:, :, 1]  # 過去の p(t) の系列
    if p_seq.size(1) >= 3:
        smoothness = ((p_seq[:, 2:] - 2 * p_seq[:, 1:-1] + p_seq[:, :-2]) ** 2).mean()
    else:
        smoothness = torch.tensor(0.0, device=x_pred.device)

    return mse + 0.1 * smoothness


# utils/loss.py
# import torch
# import torch.nn as nn



# def custom_loss(x_pred, x_true, p_seq):
#     """
#     x_pred: (batch, output_size) — 出力 [x(t+1), p(t+1)]
#     x_true: (batch, output_size) — 正解 [x(t+1), p(t+1)]
#     p_seq:  (batch, seq_len)     — 過去の p(t)
    
#     Returns:
#         total_loss: MSE(x) + smoothness(p)
#     """
#     # x(t+1) の予測誤差
#     mse = nn.MSELoss()(x_pred[:, 0], x_true[:, 0])

#     # p(t) の平滑性（2階差分）  jerk 最小の形
#     if p_seq.size(1) >= 3:
#         smoothness = ((p_seq[:, 2:] - 2 * p_seq[:, 1:-1] + p_seq[:, :-2]) ** 2).mean()
#     else:
#         smoothness = torch.tensor(0.0, device=x_pred.device)

#     return mse + 0.1 * smoothness
