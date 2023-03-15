from siphon.ncss import NCSS
from datetime import datetime

URL = "https://cida.usgs.gov/thredds/ncss/stageiv_combined/"
ncss = NCSS(URL)
query = ncss.query()
query.time_range(datetime(2002, 1, 1, 0, 0, 0), datetime(2002, 1, 1, 0, 0, 1)).strides(spatial=100)
query.accept("netcdf")
query.variables('Total_precipitation_surface_1_Hour_Accumulation')
data = ncss.get_data(query)
