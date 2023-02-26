import netCDF4 as nc


def main():
    ds = nc.Dataset(
        "/sci/labs/efratmorin/ranga/FloodMLRan/data/hydro.engr.scu.edu/files/gridded_obs/daily/ncfiles_2010/nldas_met_update.obs.daily.pr.1986.nc.gz")


if __name__ == "__main__":
    main()
