import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=6, num_layers=1):
        """
        LSTMベースの時系列予測モデル
        - input_size: 入力次元数（x, vx, ax, p, v, a → 6）
        - hidden_size: LSTM内部の隠れ状態の次元数
        - output_size: 出力次元数（x, vx, ax, p, v, a → 6）
        - num_layers: LSTM層のスタック数
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        - x: (batch_size, seq_len, input_size)
        - 戻り値: (batch_size, output_size)
        """
        out, _ = self.lstm(x)              # out: (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]        # 最後の時刻ステップのみ使用
        return self.fc(last_hidden)
