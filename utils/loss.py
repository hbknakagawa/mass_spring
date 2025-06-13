# utils/loss.py
import torch
import torch.nn as nn

def custom_loss(x_pred, x_true, p_seq):
    """
    x_pred: (batch, output_size) — 出力 [x(t+1), p(t+1)]
    x_true: (batch, output_size) — 正解 [x(t+1), p(t+1)]
    p_seq:  (batch, seq_len)     — 過去の p(t)
    
    Returns:
        total_loss: MSE(x) + smoothness(p)
    """
    # x(t+1) の予測誤差
    mse = nn.MSELoss()(x_pred[:, 0], x_true[:, 0])

    # p(t) の平滑性（2階差分）  jerk 最小の形
    if p_seq.size(1) >= 3:
        smoothness = ((p_seq[:, 2:] - 2 * p_seq[:, 1:-1] + p_seq[:, :-2]) ** 2).mean()
    else:
        smoothness = torch.tensor(0.0, device=x_pred.device)

    return mse + 0.1 * smoothness
