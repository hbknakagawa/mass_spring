import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, input_data, target_data):
        """
        Parameters:
            input_data (np.ndarray): 入力データ [N, seq_len, input_size]
            target_data (np.ndarray): 出力データ [N, output_size]
        """
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        x = self.input_data[idx]
        y = self.target_data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    