import cdsapi


def download_ERA5_one_year(client, year):
    client.retrieve(
        'reanalysis-era5-land',
        {
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
            'year': f'{year}',
            'variable': 'total_precipitation',
            'format': 'netcdf',
            'area': [
                90, 40, 40,
                160,
            ]
        },
        f'../data/ERA5/Precipitation/tp_CA_{year}.nc')


def main():
    c = cdsapi.Client()
    for year in range(1981, 2023):
        download_ERA5_one_year(c, year)


if __name__ == "__main__":
    main()
