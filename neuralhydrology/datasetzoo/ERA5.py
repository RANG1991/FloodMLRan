from pathlib import Path
from typing import List, Dict, Union
import logging
import pandas as pd
import xarray
import numpy as np

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config

logger = logging.Logger("ERA5")


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
        df_forcings = load_ERA5_forcings(data_dir=self.cfg.data_dir, basin=basin)
        df_discharge = load_ERA5_discharge(data_dir=self.cfg.data_dir, basin=basin)
        df_forcings = df_forcings.fillna(np.nan)
        df_forcings.loc[df_forcings["precip"] < 0, "precip"] = np.nan
        df_discharge = df_discharge.fillna(np.nan)
        df_discharge.loc[df_discharge["flow"] < 0, "flow"] = np.nan
        if df_forcings.empty or df_discharge.empty:
            return pd.DataFrame(columns=["date", "precip", "flow"])
        df_forcings = df_forcings.merge(df_discharge, on="date")
        df_forcings = df_forcings.set_index("date")
        return df_forcings

    def _load_attributes(self) -> pd.DataFrame:
        """Load dataset attributes

        This function is used to load basin attribute data (e.g. CAMELS catchments attributes) as a basin-indexed
        dataframe with features in columns.

        Returns
        -------
        pd.DataFrame
            Basin-indexed DataFrame, containing the attributes as columns.
        """
        df_attr = load_ERA5_attributes(data_dir=self.cfg.data_dir)
        return df_attr


def load_ERA5_forcings(data_dir: Path, basin: str) -> pd.DataFrame:
    """

    Parameters
    ----------
    data_dir
    basin

    Returns
    -------

    """
    forcing_path = data_dir / 'ERA5/ERA_5_all_data'
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")

    file_path = list(forcing_path.glob(f'**/data24_{basin}.csv'))
    if file_path:
        file_path = file_path[0]
    else:
        return pd.DataFrame(columns=["date", "precip"])

    with open(file_path, 'r') as fp:
        df = pd.read_csv(fp, sep=',')
        df["date"] = pd.to_datetime(df.date, format="%Y-%m-%d")
        df = df[["date", "precip"]]
    return df


def load_ERA5_discharge(data_dir: Path, basin: str) -> pd.DataFrame:
    """

    Parameters
    ----------
    data_dir
    basin

    Returns
    -------

    """
    discharge_path = data_dir / 'ERA5/ERA_5_all_data'
    file_path = list(discharge_path.glob(f'**/data24_{basin}.csv'))
    if file_path:
        file_path = file_path[0]
    else:
        return pd.DataFrame(columns=["date", "flow"])

    with open(file_path, 'r') as fp:
        df = pd.read_csv(fp, sep=',')
        df["date"] = pd.to_datetime(df.date, format="%Y-%m-%d")
        df = df[["date", "flow"]]
    return df


def load_ERA5_attributes(data_dir: Path) -> pd.DataFrame:
    """Load CAMELS US attributes from the dataset provided by [#]_

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US directory. This folder must contain a 'camels_attributes_v2.0' folder (the original
        data set) containing the corresponding txt files for each attribute group.
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of all basins
        are returned.

    Returns
    -------
    pandas.DataFrame
        Basin-indexed DataFrame, containing the attributes as columns.

    References
    ----------
    .. [#] Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P.: The CAMELS data set: catchment attributes and
        meteorology for large-sample studies, Hydrol. Earth Syst. Sci., 21, 5293-5313, doi:10.5194/hess-21-5293-2017,
        2017.
    """
    attributes_path_caravan = data_dir / 'ERA5/attributes_caravan_us.csv'
    attributes_path_hydroatlas = data_dir / 'ERA5/attributes_hydroatlas_us.csv'

    if not attributes_path_caravan.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path_caravan}")

    if not attributes_path_hydroatlas.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path_hydroatlas}")

    df_attr_caravan = pd.read_csv(attributes_path_caravan, dtype={'gauge_id': str})
    df_attr_hydroatlas = pd.read_csv(attributes_path_hydroatlas, dtype={'gauge_id': str})
    df_attr = df_attr_caravan.merge(df_attr_hydroatlas, on="gauge_id")
    df_attr = df_attr.set_index('gauge_id')
    df_attr['gauge_id'] = df_attr['gauge_id'].apply(lambda x: str(x).replace("us_", ""))
    return df_attr
