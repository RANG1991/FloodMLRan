import pandas as pd


def read_data_from_ERA5_dataset():



def read_data_from_caravan_dataset():
    df = pd.read_csv("C:\\Users\\galun\\Desktop\\Caravan\\timeseries\\"
                     "csv\\us\\us_01052500.csv")
    df = df[["date", "total_precipitation_sum", "potential_evaporation_sum", "streamflow"]]



def main():
