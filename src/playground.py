import pygrib

grbs = pygrib.open('src/ST4.2010103123.01h.Z')
grb = grbs.read()
print(grb.values)
