import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=6, num_layers=1):
        """
        シンプルなRNNベースの時系列予測モデル
        - input_size: 入力次元数（x, vx, ax, p, v, a → 6）
        - hidden_size: RNN内部の隠れ状態の次元数
        - output_size: 出力次元数（次の x, vx, ax, p, v, a → 6）
        - num_layers: RNN層のスタック数
        """
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        - x: (batch_size, seq_len, input_size)
        - 戻り値: (batch_size, output_size)
        """
        out, _ = self.rnn(x)              # out: (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]       # 最後の時刻ステップのみ使用

        x_next = self.fc(last_hidden)
        return torch.sigmoid(self.fc(last_hidden)) * 1.0 + 1.0  # [1.0, 2.0] の範囲に制約
