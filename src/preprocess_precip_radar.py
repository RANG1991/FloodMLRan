import metpy
from siphon.catalog import TDSCatalog
from datetime import datetime, timedelta

# Set up access via NCSS
gfs_catalog = "https://cida.usgs.gov/thredds/catalog.xml?dataset=USGS THREDDS Holdings/National Stage IV Quantitative " \
              "Precipitation Estimate Mosaic"
cat = TDSCatalog(gfs_catalog)
dataset = cat.datasets["National Stage IV Quantitative Precipitation Estimate Mosaic"]
