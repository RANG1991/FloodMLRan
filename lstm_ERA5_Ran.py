import torch
from os import listdir
from os.path import isfile, join
import xarray as xr

TEST_PERIOD = ("1989-10-01", "1999-09-30")
TRAINING_PERIOD = ("1999-10-01", "2008-09-30")


class LSTM_ERA5(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, sequence_length):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.lstm = torch.nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                                  bias=True, batch_first=True)
        self.fc = torch.nn.Linear(in_features=self.hidden_dim, out_features=1)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return self.fc(h_n)


def read_basin_file(basin_filename):
    basin_info_file = xr.load_dataset(basin_filename)
    print(basin_info_file.variables.keys())


def preprocess_data(basins_files_dir, num_basins):
    basins_files = [f for f in listdir(basins_files_dir) if isfile(join(basins_files_dir, f))]
    for i in range(min(num_basins, len(basins_files))):
        basin_filename = join(basins_files_dir, basins_files[i])
        read_basin_file(basin_filename)


def main():
    preprocess_data("./data/ERA-5/", 1)


if __name__ == "__main__":
    main()
