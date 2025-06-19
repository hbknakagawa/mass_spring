# utils/trainer.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from utils.dataset import SequenceDataset
from utils.loss import custom_loss

def train_model(model_class, input_size, output_size, hidden_size,
                seq_len, batch_size, num_epochs, learning_rate,
                save_interval, patience):

    # 保存ディレクトリ作成
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    log_dir = f"logs/learning_logs/{timestamp}"
    os.makedirs(f"{log_dir}/models", exist_ok=True)

    # データ読み込み
    input_files = sorted([f for f in os.listdir("processed_datasets") if f.startswith("input_")])
    target_files = sorted([f for f in os.listdir("processed_datasets") if f.startswith("target_")])
    input_data = [np.load(os.path.join("processed_datasets", f)) for f in input_files]
    target_data = [np.load(os.path.join("processed_datasets", f)) for f in target_files]

    # train/test 分割
    split = int(len(input_data) * 0.8)
    train_inputs = np.concatenate(input_data[:split])
    train_targets = np.concatenate(target_data[:split])
    test_inputs = np.concatenate(input_data[split:])
    test_targets = np.concatenate(target_data[split:])

    # データローダー
    train_loader = DataLoader(SequenceDataset(train_inputs, train_targets), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(SequenceDataset(test_inputs, test_targets), batch_size=batch_size, shuffle=False)

    # モデル・最適化
    model = model_class(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = custom_loss(output, y_batch, x_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 検証
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                output = model(x_batch)
                loss = custom_loss(output, y_batch, x_batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch}/{num_epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

        # モデル保存
        if epoch % save_interval == 0:
            torch.save(model.state_dict(), f"{log_dir}/models/model_{epoch}.pth")

        # # Early stopping
        # if avg_val_loss < best_loss:
        #     best_loss = avg_val_loss
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print("Early stopping triggered.")
        #         break

    # 損失プロット保存
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Training and Validation Loss')
    plt.savefig(f"{log_dir}/loss_curve.png")
    plt.close()

    # CSV保存
    pd.DataFrame({
        'epoch': list(range(1, len(train_losses)+1)),
        'train_loss': train_losses,
        'val_loss': val_losses
    }).to_csv(f"{log_dir}/progress.csv", index=False)

    print(f"Model and Logs Saved {log_dir}")
