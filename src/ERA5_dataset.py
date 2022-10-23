from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import numpy as np
import pandas as pd
import torch


class Dataset_ERA5(Dataset):

    def __init__(self, data_file, is_training=True, sequence_length=270):
        self.is_training = is_training
        self.sequence_length = sequence_length
        self.data_file = data_file
        self.X_data = np.array([])
        self.y_data = np.array([])
        self.preprocess_data()

    def __len__(self):
        return max([0, self.X_data.shape[0] - self.sequence_length])

    def __getitem__(self, index) -> T_co:
        X_data_tensor = torch.tensor(self.X_data[index: index + self.sequence_length])
        y_data_tensor = torch.tensor(self.y_data[index + self.sequence_length])
        return X_data_tensor, y_data_tensor

    def preprocess_data(self):
        df_data = pd.read_csv(self.data_file)
        print(df_data.head())
        df_data = df_data.dropna()
        self.X_data = df_data["precip"].to_numpy()
        self.X_data -= np.mean(self.X_data)
        self.X_data /= np.std(self.X_data)
        self.y_data = df_data["flow"].to_numpy()
        self.X_data = self.X_data.reshape(-1, 1).astype(np.float32)
        self.y_data = self.y_data.reshape(-1, 1).astype(np.float32)
