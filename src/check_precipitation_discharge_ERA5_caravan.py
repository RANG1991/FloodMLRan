import pandas as pd
import matplotlib.pyplot as plt


def read_data_from_ERA5_dataset():
    df_precip = pd.read_csv("C:\\Users\\galun\\Desktop\\precip24_01031500.csv")
    df_precip = df_precip[["date", "precip"]]
    df_flow = pd.read_csv("C:\\Users\\galun\\Desktop\\dis24_01031500.csv")
    df_flow = df_flow[["date", "flow"]]
    df = df_precip.merge(df_flow, on="date")
    df = df.rename(columns={"precip": "precip_ERA5", "flow": "flow_ERA5"})
    return df


def read_data_from_caravan_dataset():
    df = pd.read_csv("C:\\Users\\galun\\Desktop\\Caravan\\timeseries\\"
                     "csv\\us\\us_01031500.csv")
    df = df[["date", "total_precipitation_sum", "streamflow"]]
    df = df.rename(columns={"total_precipitation_sum": "precip_caravan", "streamflow": "flow_caravan"})
    return df


def main():
    df_ERA5 = read_data_from_ERA5_dataset()
    df_caravan = read_data_from_caravan_dataset()
    df_precip_merged = df_ERA5[["date", "precip_ERA5"]].merge(df_caravan[["date", "precip_caravan"]], on="date").iloc[:50]
    df_flow_merged = df_ERA5[["date", "flow_ERA5"]].merge(df_caravan[["date", "flow_caravan"]], on="date").iloc[:50]
    df_precip_merged.plot(x="date")
    df_flow_merged.plot(x="date")
    plt.show()


if __name__ == '__main__':
    main()

