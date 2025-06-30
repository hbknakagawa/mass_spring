import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from datetime import datetime
import matplotlib.pyplot as plt  # ★ 追加
import pandas as pd              # ★ 追加

# ==== RNN Model ====
class RNNPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(RNNPredictor, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        last_out = out[:, -1, :]  # 最後の時刻ステップを取得
        return self.fc(last_out)

# ==== Load Dataset ====
def load_dataset(data_dir):
    input_files = sorted([f for f in os.listdir(data_dir) if f.startswith("input")])
    target_files = sorted([f for f in os.listdir(data_dir) if f.startswith("target")])

    inputs = []
    targets = []
    for in_file, tgt_file in zip(input_files, target_files):
        x = np.load(os.path.join(data_dir, in_file))
        y = np.load(os.path.join(data_dir, tgt_file))
        inputs.append(x)
        targets.append(y)

    X = np.concatenate(inputs, axis=0)
    Y = np.concatenate(targets, axis=0)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# ==== Training Loop with Saving ====
def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-3, save_path=None):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "models"), exist_ok=True)

    train_losses = []  # ★ 追加
    val_losses = []    # ★ 追加

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                y_pred = model(x_val)
                val_loss += criterion(y_pred, y_val).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)  # ★
        val_losses.append(avg_val_loss)      # ★

        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # モデル保存
        if epoch % 100 == 0 or epoch == num_epochs:
            torch.save(model.state_dict(), os.path.join(save_path, "models", f"model_epoch{epoch}.pth"))

    # ★ ロス曲線の保存
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()

    # ★ ロス値CSV保存
    df = pd.DataFrame({
        "epoch": list(range(1, num_epochs + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses
    })
    df.to_csv(os.path.join(save_path, "progress.csv"), index=False)

# ==== Main ====
def main():
    data_dir = "processed_datasets"
    X, Y = load_dataset(data_dir)

    dataset = TensorDataset(X, Y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    input_dim = X.shape[2]  # 特徴量の次元
    output_dim = Y.shape[1]  # 出力の次元
    hidden_dim = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = RNNPredictor(input_dim, hidden_dim, output_dim).to(device)

    # 保存パスの作成
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join("logs", "learning_logs", timestamp)

    train_model(model, train_loader, val_loader, num_epochs=1000, lr=1e-3, save_path=save_path)

if __name__ == "__main__":
    main()
