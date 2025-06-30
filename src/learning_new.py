# src/learning.py

from model.RNN import RNNModel  # または from models.LSTM import LSTMModel
from utils.trainer import train_model

# tensorboard --logdir=logs/learning_logs
#  http://localhost:6006 



# === ハイパーパラメータ設定 ===
INPUT_SIZE = 2         # 特徴量 [x(t), p(t)]
OUTPUT_SIZE = 1       # 出力 [x(t+1)]
HIDDEN_SIZE = 64
SEQ_LEN = 10
BATCH_SIZE = 16
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-3
SAVE_INTERVAL = 100
PATIENCE = 10

# === 学習実行 ===
if __name__ == "__main__":
    train_model(
        model_class=RNNModel,         # または LSTMModel に変更可
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        save_interval=SAVE_INTERVAL,
        patience=PATIENCE
    )
