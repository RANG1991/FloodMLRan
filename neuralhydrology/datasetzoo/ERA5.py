from pathlib import Path
from typing import Tuple, List, Dict, Union

import pandas as pd
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class ERA5(BaseDataset):
    """Template class for adding a new data set.

    Each dataset class has to derive from `BaseDataset`, which implements most of the logic for preprocessing data and
    preparing data for model training. Only two methods have to be implemented for each specific dataset class:
    `_load_basin_data()`, which loads the time series data for a single basin, and `_load_attributes()`, which loads
    the static attributes for the specific data set.

    Usually, we outsource the functions to load the time series and attribute data into separate functions (in the
    same file), which we then call from the corresponding class methods. This way, we can also use specific basin data
    or dataset attributes without these classes.

    To make this dataset available for model training, don't forget to add it to the `get_dataset()` function in
    'neuralhydrology.datasetzoo.__init__.py'
    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used.
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset, and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        # initialize parent class
        super(ERA5, self).__init__(cfg=cfg,
                                   is_train=is_train,
                                   period=period,
                                   basin=basin,
                                   additional_features=additional_features,
                                   id_to_int=id_to_int,
                                   scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load basin time series data

        This function is used to load the time series data (meteorological forcing, streamflow, etc.) and make available
        as time series input for model training later on. Make sure that the returned dataframe is time-indexed.

        Parameters
        ----------
        basin : str
            Basin identifier as string.
        Returns
        -------
        pd.DataFrame
            Time-indexed DataFrame, containing the time series data (e.g., forcings + discharge).
        """
        return load_ERA5_forcings(data_dir=self.cfg.data_dir, basin=basin)

    def _load_attributes(self) -> pd.DataFrame:
        """Load dataset attributes

        This function is used to load basin attribute data (e.g. CAMELS catchments attributes) as a basin-indexed
        dataframe with features in columns.

        Returns
        -------
        pd.DataFrame
            Basin-indexed DataFrame, containing the attributes as columns.
        """
        return pd.DataFrame()


def load_ERA5_forcings(data_dir: Path, basin: str) -> pd.DataFrame:
    """Load the forcing data for a basin of the CAMELS US data set.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US directory. This folder must contain a 'basin_mean_forcing' folder containing one
        subdirectory for each forcing. The forcing directories have to contain 18 subdirectories (for the 18 HUCS) as in
        the original CAMELS data set. In each HUC folder are the forcing files (.txt), starting with the 8-digit basin
        id.
    basin : str
        8-digit USGS identifier of the basin.
    forcings : str
        Can be e.g. 'daymet' or 'nldas', etc. Must match the folder names in the 'basin_mean_forcing' directory.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the forcing data..
    """
    forcing_path = data_dir / 'ERA5/all_ERA5_data'
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")

    file_path = list(forcing_path.glob(f'**/data24_{basin}.csv'))
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {file_path}')

    with open(file_path, 'r') as fp:
        df = pd.read_csv(fp, sep=',')
        df["date"] = pd.to_datetime(df.date, format="%Y-%m-%d")
        df = df.set_index("date")
        df = df[["date", "precip"]]
    return df


def load_ERA5_discharge(data_dir: Path, basin: str) -> pd.Series:
    """Load the discharge data for a basin of the CAMELS US data set.

    Parameters
    ----------
    discharge
    data_dir : Path
        Path to the CAMELS US directory. This folder must contain a 'usgs_streamflow' folder with 18
        subdirectories (for the 18 HUCS) as in the original CAMELS data set. In each HUC folder are the discharge files
        (.txt), starting with the 8-digit basin id.
    basin : str
        8-digit USGS identifier of the basin.

    Returns
    -------
    pd.Series
        Time-index pandas.Series of the discharge values (mm/day)
    """

    discharge_path = data_dir / 'ERA5/all_ERA5_data'
    file_path = list(discharge_path.glob(f'**/data24_{basin}.csv'))
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {file_path}')

    with open(file_path, 'r') as fp:
        df = pd.read_csv(fp, sep=',')
        df["date"] = pd.to_datetime(df.date, format="%Y-%m-%d")
        df = df.set_index("date")
        df = df[["date", "flow"]]
    return df.flow
