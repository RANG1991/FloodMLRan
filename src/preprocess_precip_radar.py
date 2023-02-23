import urllib.request

url = """https://cida.usgs.gov/thredds/ncss/stageiv_combined?var=Total_precipitation_surface_1_Hour_Accumulation
&disableProjSubset=on&horizStride=1&time_start=2002-01-01T00%3A00%3A00Z&time_end=2002-01-02T21%3A00%3A00Z&timeStride
=1"""

with urllib.request.urlopen(url.replace("\n", "").replace(" ", "")) as f:
    html = f.read().decode('utf-8')
