from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import numpy as np
import pandas as pd


class Dataset_ERA5(Dataset):

    def __init__(self, X_data_file, is_training=True, sequence_length=270):
        self.is_training = is_training
        self.sequence_length = sequence_length
        self.X_data_file = X_data_file
        self.X_data = np.array([])
        self.y_data = np.array([])
        self.preprocess_data()

    def __len__(self):
        return max([0, self.X_data.shape[0] - self.sequence_length])

    def __getitem__(self, index) -> T_co:
        return self.X_data[index: index + self.sequence_length, :]

    def preprocess_data(self):
        self.X_data = np.fromfile(self.X_data_file)
        self.X_data -= np.mean(self.X_data)
        self.X_data /= np.std(self.X_data)
        self.y_data = np.array([])
